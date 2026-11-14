"""Leader-follower teleop between two YAM arms in one process, no teaching handle needed.

The leader floats in gravity compensation; move it by hand. Press Enter to start following: the
follower eases to the leader's pose over ``--ramp-s`` seconds, then tracks it with its normal PD
gains. Press Enter again to stop; the follower holds where it is. Ctrl+C closes both arms (they go
limp ~400 ms later -- support them).

Each arm needs its own CAN adapter. macOS (gs_usb, sudo to claim the USB devices):
    sudo .venv/bin/python scripts/teleop_leader_follower.py \\
        --leader-channel gsusb:<leader serial> --follower-channel gsusb:<follower serial>

Dry run with no hardware:
    python scripts/teleop_leader_follower.py --sim
"""

import logging
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmName, ArmType, GripperType


@dataclass
class Args:
    leader_channel: str = "can0"
    """Leader arm's CAN channel: a SocketCAN name, or gsusb:<adapter serial> on macOS."""
    follower_channel: str = "can1"
    """Follower arm's CAN channel."""
    leader_arm: ArmName = "yam"
    follower_arm: ArmName = "yam"
    leader_gripper: str = "linear_4310"
    """End effector on the leader; only its mass matters (for gravity compensation)."""
    follower_gripper: str = "linear_4310"
    mirror_gripper: bool = False
    """Copy the leader's gripper opening to the follower. Off: the follower's gripper holds still."""
    rate_hz: float = 100.0
    """How often the follower is sent the leader's pose."""
    ramp_s: float = 3.0
    """Seconds the follower takes to ease onto the leader's pose when following starts."""
    max_joint_speed: float = 2.0
    """Cap on how fast any follower arm joint is commanded to move (rad/s), whatever the leader does."""
    sim: bool = False
    """Use simulated arms instead of hardware."""


def _toggle_on_enter(following: threading.Event, stop: threading.Event) -> None:
    """Flip ``following`` each time Enter is pressed."""
    while not stop.is_set():
        if sys.stdin.readline() == "":  # stdin closed
            return
        if following.is_set():
            following.clear()
        else:
            following.set()


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


def run(args: Args, leader: Robot, follower: Robot, n_arm: int) -> None:
    n = follower.num_dofs() # number of dofs in follower arm
    has_gripper = n > n_arm  # anything pastwthe arm joints is the gripper
    period = 1.0 / args.rate_hz
    max_step = args.max_joint_speed * period

    following = threading.Event()
    stop = threading.Event()
    threading.Thread(target=_toggle_on_enter, args=(following, stop), daemon=True).start()
    print("Ready. Enter = start/stop following, Ctrl+C = quit.", flush=True)

    was_following = False
    command = follower.get_joint_pos().copy()
    origin = command.copy()  # follower pose when following started: the ease-in blends from here
    gripper_hold = command[n_arm:].copy()
    ramp_start = 0.0
    halted = False

    try:
        while True:
            tick = time.monotonic()
            halted, command = _halt_following_when_an_arm_dies(leader, follower, following, halted, command)
            if halted:
                time.sleep(period)
                continue
            now_following = following.is_set()
            if now_following and not was_following:
                command, origin, gripper_hold, ramp_start = _start_following(follower, tick, n_arm)
            elif was_following and not now_following:
                command = _stop_following(follower)
            was_following = now_following

            if was_following:
                target = command.copy()
                target[:n_arm] = leader.get_joint_pos()[:n_arm]
                if has_gripper:
                    target[n_arm:] = leader.get_joint_pos()[n_arm:n] if args.mirror_gripper else gripper_hold
                blend = min(1.0, (tick - ramp_start) / args.ramp_s) if args.ramp_s > 0 else 1.0
                target[:n_arm] = origin[:n_arm] + blend * (target[:n_arm] - origin[:n_arm])
                command = _step_toward(command, target, max_step, n_arm)
                follower.command_joint_pos(command)

            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    finally:
        stop.set()


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    if args.mirror_gripper and args.leader_gripper != args.follower_gripper:
        raise SystemExit("--mirror-gripper needs the same gripper type on both arms")

    def build(channel: str, arm: str, gripper: str) -> Robot:
        return get_yam_robot(
            channel=channel,
            arm_type=ArmType.from_string_name(arm),
            gripper_type=GripperType.from_string_name(gripper),
            sim=args.sim,
        )

    leader = build(args.leader_channel, args.leader_arm, args.leader_gripper)
    try:
        follower = build(args.follower_channel, args.follower_arm, args.follower_gripper)
    except BaseException:
        leader.close()
        raise
    try:
        def arm_joints(robot: Robot, gripper: str) -> int:
            g = GripperType.from_string_name(gripper)
            return robot.num_dofs() - (0 if g.is_teaching_handle or g == GripperType.NO_GRIPPER else 1)
        
        follower_n_arm = arm_joints(follower, args.follower_gripper)
        leader_n_arm = arm_joints(leader, args.leader_gripper)
        if follower_n_arm != leader_n_arm:
            raise SystemExit("leader and follower must have the same number of arm joints")
        run(args, leader, follower, leader_n_arm)
    except KeyboardInterrupt:
        pass
    finally:
        follower.close()
        leader.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
