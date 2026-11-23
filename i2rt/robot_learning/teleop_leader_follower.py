"""Leader-follower teleop between two YAM arms in one process, no teaching handle needed.

The leader floats in gravity compensation; move it by hand. Press Enter (or, with a teaching handle,
its top button) to start following: the follower eases to the leader's pose over ``--ramp-s``
seconds, then tracks it with its normal PD gains. Press again to stop; the follower holds where it is. ``--bilateral-kp`` springs the
leader toward the follower's measured pose while following, so a lagging follower pushes back on
your hand. Ctrl+C closes both arms (they go limp ~400 ms later -- support them).

Pass ``--demo-root`` to record demonstrations while following. Each start-to-stop stretch, after
the ease-in, is written straight away as its own ``episode_NNNN`` folder: the overhead and
right-wrist camera frames, the follower's joints and the command sent to it (see
``episode_recorder``). Nothing is converted while recording. The two cameras are shown side by
side in a window the whole time (not under --sim-follower); ``--task`` labels the episodes.
Afterwards, turn the folder into a LeRobot dataset and train on it::

    python -m i2rt.robot_learning.convert_to_lerobot --demo-root <demo root> --dataset-root <dataset>
    python -m i2rt.robot_learning.train_lerobot --dataset-root <dataset>

Each arm needs its own CAN adapter. macOS (gs_usb, sudo to claim the USB devices):
    sudo .venv/bin/python -m i2rt.robot_learning.teleop_leader_follower \\
        --leader-channel gsusb:<leader serial> --follower-channel gsusb:<follower serial>

Dry run with no hardware:
    python -m i2rt.robot_learning.teleop_leader_follower --sim

Real leader driving a simulated follower (the window shows the follower; Enter is still
typed in this terminal). On macOS the window needs mjpython:
    sudo .venv/bin/mjpython -m i2rt.robot_learning.teleop_leader_follower \\
        --sim-follower --leader-channel gsusb:<leader serial>


Currently, the CAN serial numbers are [leader arm: '20A1376F4546500B2', follower arm: '2091377E4546500A2']
"""

import logging
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmName, ArmType, GripperType, _load_arm_config


@dataclass
class Args:
    leader_channel: str = "can0"
    """Leader arm's CAN channel: a SocketCAN name, or gsusb:<adapter serial> on macOS."""
    follower_channel: str = "can1"
    """Follower arm's CAN channel."""
    leader_arm: ArmName = "yam"
    follower_arm: ArmName = "yam"
    leader_gripper: str = "yam_teaching_handle_v1"
    """End effector on the leader. A teaching handle's trigger is copied to the follower gripper."""
    follower_gripper: str = "linear_4310"
    mirror_gripper: bool = False
    """Copy a motorized leader gripper to the follower. A teaching handle trigger is copied either way."""
    rate_hz: float = 100.0
    """How often the follower is sent the leader's pose."""
    ramp_s: float = 3.0
    """Seconds the follower takes to ease onto the leader's pose when following starts."""
    max_joint_speed: float = 2.0
    """Cap on how fast any follower arm joint is commanded to move (rad/s), whatever the leader does."""
    bilateral_kp: float = 0.0
    """While following, fraction of the leader's position gain that pulls its arm joints toward the
    follower's measured pose. 0 leaves the leader floating. The gripper is not sprung."""
    sim: bool = False
    """Simulate both arms. No CAN connection and no window."""
    sim_follower: bool = False
    """Keep the real leader and simulate the follower, drawn in a MuJoCo window. On macOS, run this with mjpython."""
    demo_root: str = ""
    """Directory the demonstrations are recorded into, one folder per episode. Empty skips recording."""
    record_fps: int = 30
    """Frames saved per second. The arms keep running at --rate-hz."""
    task: str = "teleop"
    """What the demonstration is, stored with every episode."""
    overhead_index: int = 0
    """Device index of the overhead camera, used only while recording."""
    overhead_orientation: Literal["ROTATE_0", "ROTATE_90", "ROTATE_180", "ROTATE_270"] = "ROTATE_0"
    right_wrist_index: int = 1
    """Device index of the right wrist camera, used only while recording."""
    right_wrist_orientation: Literal["ROTATE_0", "ROTATE_90", "ROTATE_180", "ROTATE_270"] = "ROTATE_180"


def _toggle_on_enter(following: threading.Event, stop: threading.Event) -> None:
    """Flip ``following`` each time Enter is pressed."""
    while not stop.is_set():
        if sys.stdin.readline() == "":  # stdin closed
            return
        if following.is_set():
            following.clear()
        else:
            following.set()


class _SpaceHeld:
    """Whether the space bar is being held, from a global keyboard listener (pynput).

    A terminal only ever sees typed characters, never a key being released, so holding a key needs
    a listener; on macOS that needs Input Monitoring permission for the terminal app. Without
    pynput or the permission, ``held`` simply stays False and everything else works as before.
    """

    def __init__(self) -> None:
        self._held = False
        self._listener: Any = None
        try:
            from pynput import keyboard
        except ImportError:
            print("Space-bar following is off: pynput is not installed (uv pip install pynput).", flush=True)
            return
        space = keyboard.Key.space

        def on_press(key: Any) -> None:
            if key == space:
                self._held = True

        def on_release(key: Any) -> None:
            if key == space:
                self._held = False

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()
        if getattr(keyboard.Listener, "IS_TRUSTED", True) is False:
            print(
                "Space-bar following may not work: allow this terminal under System Settings > Privacy & "
                "Security > Input Monitoring, then restart it.",
                flush=True,
            )

    @property
    def held(self) -> bool:
        return self._held

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()


TOP_BUTTON = 0
"""``io_inputs`` index of the handle's top button (start/stop following), as in examples/minimum_gello."""
BOTTOM_BUTTON = 1
"""``io_inputs`` index of the handle's second button (delete the last recorded episode)."""


def _button_down(leader: Robot, button: int) -> bool:
    """Whether a teaching-handle button is held. False with no handle or no reading yet.

    The chain reads the handle every control cycle; ``io_inputs`` holds its two buttons.
    """
    chain = getattr(leader, "motor_chain", None)
    read = getattr(chain, "get_same_bus_device_states", None)
    states = read() if read is not None else None
    if not states or len(states[0].io_inputs) <= button:
        return False
    return bool(states[0].io_inputs[button] > 0.5)


def _alive(robot: Robot) -> bool:
    """False once a real arm's control loop has stopped (it then keeps reporting its last pose)."""
    chain = getattr(robot, "motor_chain", None)
    return chain is None or bool(getattr(chain, "running", True))


def _step_toward(current: np.ndarray, target: np.ndarray, max_step: float, n_arm: int) -> np.ndarray:
    """``target``, with each arm joint moved at most ``max_step`` from ``current``; gripper uncapped."""
    out = target.copy()
    out[:n_arm] = current[:n_arm] + np.clip(target[:n_arm] - current[:n_arm], -max_step, max_step)
    return out


def _halt_following_when_an_arm_dies(
    leader: Robot,
    follower: Robot,
    following: threading.Event,
    halted: bool,
    command: np.ndarray,
) -> tuple[bool, np.ndarray]:
    dead = [name for name, robot in (("leader", leader), ("follower", follower)) if not _alive(robot)]
    if dead and not halted:
        halted = True
        following.clear()
        if "follower" not in dead:
            command = follower.get_joint_pos().copy()
            follower.command_joint_pos(command)
    if halted:
        following.clear()
    return halted, command


def _start_following(follower: Robot, tick: float, n_arm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    print("Following ON: easing onto the leader...", flush=True)
    command = follower.get_joint_pos().copy()
    return command, command.copy(), command[n_arm:].copy(), tick


def _stop_following(follower: Robot) -> np.ndarray:
    print("Following OFF: follower holding.", flush=True)
    command = follower.get_joint_pos().copy()
    follower.command_joint_pos(command)
    return command


def _feedback_gains(base_kp: np.ndarray, bilateral_kp: float, n_arm: int) -> tuple[np.ndarray, np.ndarray]:
    kp = np.zeros_like(base_kp)
    kp[:n_arm] = base_kp[:n_arm] * bilateral_kp
    return kp, np.zeros_like(base_kp)


def _pull_leader_toward_follower(
    leader: MotorChainRobot,
    follower: Robot,
    n_arm: int,
    kp: np.ndarray,
    kd: np.ndarray,
) -> None:
    leader.update_kp_kd(kp.copy(), kd.copy())
    command = leader.get_joint_pos().copy()
    command[:n_arm] = follower.get_joint_pos()[:n_arm]
    leader.command_joint_pos(command)


def _float_leader(leader: MotorChainRobot, kp: np.ndarray, kd: np.ndarray) -> None:
    leader.update_kp_kd(kp.copy(), kd.copy())
    leader.enter_gravity_comp_idle()


def _leader_gripper_opening(leader: Robot, n_arm: int, n_gripper: int) -> np.ndarray | None:
    """Gripper command in the follower's joint vector, or None when the leader has no reading yet.

    A motorized gripper is the tail of ``get_joint_pos``. A teaching handle is not a joint: its
    trigger is ``get_observations()["gripper_pos"]``, already 0 = closed and 1 = open, which is the
    same normalized command the follower gripper takes.
    """
    pos = leader.get_joint_pos()
    if pos.shape[0] >= n_arm + n_gripper:
        return pos[n_arm : n_arm + n_gripper]
    opening = leader.get_observations().get("gripper_pos")
    if opening is None:
        return None
    values = np.atleast_1d(np.asarray(opening, dtype=float))
    if values.shape[0] < n_gripper:
        return None
    return values[:n_gripper].copy()


def _require_mjpython() -> None:
    """macOS only opens the MuJoCo window from an interpreter mjpython launched."""
    if sys.platform != "darwin":
        return
    import mujoco.viewer

    if getattr(mujoco.viewer, "_MJPYTHON", None) is None:
        raise SystemExit(
            "The simulated follower opens a MuJoCo window. On macOS start this with mjpython:\n"
            "    sudo .venv/bin/mjpython -m i2rt.robot_learning.teleop_leader_follower "
            "--sim-follower --leader-channel gsusb:<leader serial>"
        )


def _sync_follower_view(viewer: Any) -> bool:
    """Draw the latest simulated pose. False once the window has been closed."""
    if viewer is None:
        return True
    if not viewer.is_running():
        return False
    viewer.sync()
    return True


def _camera_frame(
    frames: Any, recorder: Any | None, show: bool, record: bool, follower: Robot, command: np.ndarray
) -> None:
    """Show the cameras' newest frames, and add them to the current episode when ``record``."""
    from i2rt.robot_learning.cameras import CameraRole
    from i2rt.robot_learning.read_cameras import show_cameras

    overhead = frames.get(CameraRole.OVERHEAD)
    wrist = frames.get(CameraRole.RIGHT_WRIST)
    if show:
        show_cameras(overhead, wrist)
    if record and recorder is not None and overhead is not None and wrist is not None:
        recorder.add(follower.get_joint_pos(), command, overhead, wrist)


def run(
    args: Args,
    leader: Robot,
    follower: Robot,
    n_arm: int,
    recorder: Any | None = None,
    frames: Any | None = None,
    viewer: Any | None = None,
    show_cameras: bool = False,
) -> None:
    n = follower.num_dofs()  # number of dofs in follower arm
    has_gripper = n > n_arm  # anything past the arm joints is the gripper
    leader_handle = GripperType.from_string_name(args.leader_gripper).is_teaching_handle
    copy_gripper = has_gripper and (args.mirror_gripper or leader_handle)
    period = 1.0 / args.rate_hz
    max_step = args.max_joint_speed * period

    leader_hw = leader if isinstance(leader, MotorChainRobot) else None
    if args.bilateral_kp > 0 and leader_hw is None:
        raise SystemExit("--bilateral-kp needs a real motorized leader; sim has no position-gain spring")
    base_kp = np.zeros(0)
    base_kd = np.zeros(0)
    feedback_kp = np.zeros(0)
    feedback_kd = np.zeros(0)
    if leader_hw is not None and args.bilateral_kp > 0:
        info = leader_hw.get_robot_info()
        base_kp = np.array(info["kp"], dtype=float, copy=True)
        base_kd = np.array(info["kd"], dtype=float, copy=True)
        feedback_kp, feedback_kd = _feedback_gains(base_kp, args.bilateral_kp, n_arm)

    following = threading.Event()
    stop = threading.Event()
    threading.Thread(target=_toggle_on_enter, args=(following, stop), daemon=True).start()
    toggle_keys = "Enter or the handle's top button" if leader_handle else "Enter"
    print(f"Ready. {toggle_keys} = start/stop following, Ctrl+C = quit.", flush=True)
    if args.bilateral_kp > 0:
        print(f"Bilateral feedback on: leader arm kp scaled by {args.bilateral_kp}.", flush=True)
    if recorder is not None:
        print(
            f"Recording at {args.record_fps} fps once the ease-in finishes, task: {args.task}. "
            "Each start-to-stop of following is saved as its own episode."
            + (" Bottom button (between episodes) deletes the last one." if leader_handle else ""),
            flush=True,
        )
    if viewer is not None:
        print("Follower is simulated. Close its window to quit.", flush=True)
    # Holding space between episodes: follow without recording (e.g. to drive the arm back home).
    space = _SpaceHeld()
    print("Hold space (between episodes) to follow without recording.", flush=True)

    was_following = False  # follower tracking the leader: a recorded episode, or space held
    was_episode = False  # a recorded episode in progress (top button / Enter)
    command = follower.get_joint_pos().copy()
    origin = command.copy()  # follower pose when following started: the ease-in blends from here
    gripper_hold = command[n_arm:].copy()
    ramp_start = 0.0
    halted = False
    leader_sprung = False
    # Held at startup: wait for a release before acting on either button.
    top_was_down = _button_down(leader, TOP_BUTTON)
    bottom_was_down = _button_down(leader, BOTTOM_BUTTON)
    recording_failure_seen = False
    next_record = 0.0
    record_period = 1.0 / args.record_fps if args.record_fps > 0 else 0.0

    def float_leader() -> None:
        nonlocal leader_sprung
        if leader_hw is None or not leader_sprung:
            return
        _float_leader(leader_hw, base_kp, base_kd)
        leader_sprung = False

    try:
        while True:
            tick = time.monotonic()
            halted, command = _halt_following_when_an_arm_dies(leader, follower, following, halted, command)
            if halted:
                float_leader()
                if recorder is not None:
                    recorder.end_episode()
                # Keep the preview live (nothing is recorded): the window is how you see the scene.
                if frames is not None and tick >= next_record:
                    _camera_frame(frames, recorder, show_cameras, False, follower, command)
                    next_record = max(next_record + record_period, tick)
                time.sleep(period)
                if not _sync_follower_view(viewer):
                    break
                continue
            # Top button toggles following on press, like Enter; holding it down counts once.
            top_down = _button_down(leader, TOP_BUTTON)
            if top_down and not top_was_down:
                if following.is_set():
                    following.clear()
                else:
                    following.set()
            top_was_down = top_down
            # A recording failure stops following once, so the follower holding still tells you what
            # the message says; the arms stay powered and teleop can go on, unrecorded.
            if recorder is not None and recorder.failed and not recording_failure_seen:
                recording_failure_seen = True
                following.clear()
                print("*** Following stopped because recording failed. Press to keep going unrecorded.", flush=True)
            # Two ways to follow: a recorded episode (following), or space held, which records nothing.
            # During an episode space changes nothing; ending the episode with space held keeps the
            # follower following, unrecorded, until space is released.
            now_episode = following.is_set()
            now_following = now_episode or space.held
            if now_following and not was_following:
                command, origin, gripper_hold, ramp_start = _start_following(follower, tick, n_arm)
                if not now_episode:
                    print("Following without recording (space held).", flush=True)
            elif was_following and not now_following:
                command = _stop_following(follower)
                float_leader()
            if was_episode and not now_episode and recorder is not None:
                recorder.end_episode()  # instant: a background thread finishes writing it
            was_following, was_episode = now_following, now_episode

            # Bottom button deletes the episode just recorded (a bad take). Only between episodes.
            bottom_down = _button_down(leader, BOTTOM_BUTTON)
            if bottom_down and not bottom_was_down and recorder is not None:
                if was_episode or recorder.recording_episode:
                    print("Bottom button ignored: finish the episode (top button) first.", flush=True)
                else:
                    deleted = recorder.delete_last_episode()
                    if deleted is None:
                        print(
                            "Nothing to delete: no episode from this session left (earlier sessions are never deleted).",
                            flush=True,
                        )
                    else:
                        print(
                            f"Deleted {deleted}. {recorder.deletable} more from this session can be deleted.",
                            flush=True,
                        )
            bottom_was_down = bottom_down

            recording = False
            if was_following:
                target = command.copy()
                target[:n_arm] = leader.get_joint_pos()[:n_arm]
                if has_gripper:
                    opening = _leader_gripper_opening(leader, n_arm, n - n_arm) if copy_gripper else None
                    target[n_arm:] = gripper_hold if opening is None else opening
                blend = min(1.0, (tick - ramp_start) / args.ramp_s) if args.ramp_s > 0 else 1.0
                target[:n_arm] = origin[:n_arm] + blend * (target[:n_arm] - origin[:n_arm])
                command = _step_toward(command, target, max_step, n_arm)
                follower.command_joint_pos(command)
                if leader_hw is not None and args.bilateral_kp > 0:
                    _pull_leader_toward_follower(leader_hw, follower, n_arm, feedback_kp, feedback_kd)
                    leader_sprung = True
                recording = recorder is not None and was_episode and blend >= 1.0

            # Cameras run at the recording rate whether or not an episode is being recorded, so the
            # preview is live before the first press too.
            if frames is not None and tick >= next_record:
                _camera_frame(frames, recorder, show_cameras, recording, follower, command)
                # Advance by whole periods, not from this tick: the loop's 10 ms steps would otherwise
                # round every gap up (~37 ms), recording ~27 fps into episodes labelled 30.
                next_record = max(next_record + record_period, tick)

            time.sleep(max(0.0, period - (time.monotonic() - tick)))
            if not _sync_follower_view(viewer):
                break
    finally:
        stop.set()
        space.stop()
        float_leader()


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    if args.bilateral_kp < 0:
        raise SystemExit("--bilateral-kp must be >= 0")
    if args.sim and args.sim_follower:
        raise SystemExit("pass either --sim or --sim-follower")
    if args.mirror_gripper and args.leader_gripper != args.follower_gripper:
        raise SystemExit("--mirror-gripper needs the same gripper type on both arms")
    if args.sim_follower:
        _require_mjpython()

    def build(channel: str, arm: str, gripper: str, *, sim: bool) -> Robot:
        return get_yam_robot(
            channel=channel,
            arm_type=ArmType.from_string_name(arm),
            gripper_type=GripperType.from_string_name(gripper),
            sim=sim,
        )

    def joint_counts(arm: str, gripper: str) -> tuple[int, int]:
        """(arm joints, arm joints + motorized gripper), from the configs -- no robot needed."""
        n_arm = len(_load_arm_config(ArmType.from_string_name(arm)).motor_list)
        g = GripperType.from_string_name(gripper)
        return n_arm, n_arm + (0 if g.is_teaching_handle or g == GripperType.NO_GRIPPER else 1)

    leader_n_arm, _ = joint_counts(args.leader_arm, args.leader_gripper)
    follower_n_arm, follower_dofs = joint_counts(args.follower_arm, args.follower_gripper)
    if follower_n_arm != leader_n_arm:
        raise SystemExit("leader and follower must have the same number of arm joints")

    recorder = None
    cameras: dict[Any, Any] = {}
    frames = None
    # The preview is an OpenCV window, which macOS only draws from the real main thread; under
    # mjpython (--sim-follower) that thread belongs to the MuJoCo viewer.
    show = bool(args.demo_root) and not args.sim_follower
    try:
        # Everything slow happens before either arm starts. Opening cameras takes seconds and hogs the
        # GIL; done after the arms were up (then with a lerobot/torch import too), it starved both
        # control loops past the motors' 400 ms timeout and every motor faulted with "loss
        # communication" (bench, 2026-11-23).
        if args.demo_root:
            if args.record_fps <= 0:
                raise SystemExit("--record-fps must be positive")
            from i2rt.robot_learning.cameras import Camera, CameraRole, LatestFrames, Orientation
            from i2rt.robot_learning.episode_recorder import EpisodeRecorder
            from i2rt.robot_learning.lerobot_recorder import joint_names

            for role, index, orientation in (
                (CameraRole.OVERHEAD, args.overhead_index, args.overhead_orientation),
                (CameraRole.RIGHT_WRIST, args.right_wrist_index, args.right_wrist_orientation),
            ):
                camera = Camera(role, index, Orientation[orientation])
                if not camera.is_open():
                    camera.close()
                    raise SystemExit(f"{role.value} camera did not open on index {index}")
                cameras[role] = camera
            frames = LatestFrames(cameras)
            if args.sim_follower:
                print("Camera preview is off with --sim-follower (the MuJoCo window owns the display).")
            recorder = EpisodeRecorder(
                root=Path(args.demo_root),
                fps=args.record_fps,
                task=args.task,
                joint_names=joint_names(follower_dofs, follower_n_arm),
                robot_type=args.follower_arm,
            )

        leader = build(args.leader_channel, args.leader_arm, args.leader_gripper, sim=args.sim)
        try:
            follower = build(
                args.follower_channel, args.follower_arm, args.follower_gripper, sim=args.sim or args.sim_follower
            )
        except BaseException:
            leader.close()
            raise
        try:
            if follower.num_dofs() != follower_dofs:
                raise SystemExit(f"follower has {follower.num_dofs()} joints; its config says {follower_dofs}")
            viewer_cm: Any = nullcontext(None)
            if args.sim_follower:
                import mujoco.viewer

                from i2rt.robots.sim_robot import SimRobot

                if not isinstance(follower, SimRobot):
                    raise SystemExit("--sim-follower did not build a simulated follower")
                # command_joint_pos writes this same MjData, so syncing it shows the commanded pose.
                viewer_cm = mujoco.viewer.launch_passive(follower._model, follower._data)
            with viewer_cm as viewer:
                run(args, leader, follower, leader_n_arm, recorder, frames, viewer, show_cameras=show)
        except KeyboardInterrupt:
            pass
        finally:
            # Arms first: writing out the last episode's frames can take a moment; they must not wait on it.
            follower.close()
            leader.close()
    finally:
        if frames is not None:
            frames.close()
        if recorder is not None:
            recorder.close()
        for camera in cameras.values():
            camera.close()
        if show:
            from i2rt.robot_learning.read_cameras import close_camera_window

            close_camera_window()


if __name__ == "__main__":
    main(tyro.cli(Args))
