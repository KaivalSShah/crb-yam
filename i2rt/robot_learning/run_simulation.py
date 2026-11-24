"""Run a trained LeRobot policy in the simulated station with the cups and ball placed by hand.

    .venv/bin/mjpython i2rt/robot_learning/run_simulation.py        # no policy: open the scene in the MuJoCo viewer
    python i2rt/robot_learning/run_simulation.py --policy-path outputs/train/yam_act/checkpoints/last/pretrained_model

Positions are metres in the arm's frame: x forward from the base, y to the arm's left. The recorded
episodes had the cups within roughly x 0.33-0.50, y -0.20-0.20.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro

from i2rt.simulation import episodes, scene
from i2rt.simulation.env import YamStationEnv
from i2rt.simulation.run_policy import Args as PolicyArgs
from i2rt.simulation.run_policy import LeRobotPolicy, rollout


@dataclass
class Args:
    policy_path: str | None = None
    """Checkpoint folder (the ``pretrained_model`` directory ``lerobot-train`` writes) or a Hub repo id.
    Left out, no policy runs: the scene opens in MuJoCo's interactive viewer (macOS: run with mjpython)."""
    cup_orange: tuple[float, float] = (0.45, 0.12)
    cup_blue: tuple[float, float] = (0.38, -0.10)
    box: tuple[float, float] = (0.35, -0.35)
    ball_on: str = "cup_orange"
    """The cup the ball starts on; success is the ball resting on the other one."""
    max_steps: int = 450
    """Control steps (30 Hz: 450 is 15 s)."""
    device: str = "auto"
    video: Path = Path("run_simulation.mp4")


def main(args: Args) -> None:
    cups = {"cup_orange": args.cup_orange, "cup_blue": args.cup_blue}
    if args.ball_on not in cups:
        raise SystemExit(f"--ball-on must be one of {sorted(cups)}")
    target = next(cup for cup in cups if cup != args.ball_on)
    objects = {
        "cup_orange": scene.ObjectState(xy=args.cup_orange),
        "cup_blue": scene.ObjectState(xy=args.cup_blue),
        "ball": scene.ObjectState(xy=cups[args.ball_on], z=scene.ball_rest_height()),
        "box": scene.ObjectState(xy=args.box),
    }
    setup = episodes.EpisodeSetup(
        objects, source_cup=args.ball_on, target_cup=target, grasp_frame=None, release_frame=None
    )
    _, arm_start, _ = episodes.sample_start(np.random.default_rng(0))  # a recorded arm starting pose

    env = YamStationEnv()
    if args.policy_path is None:
        # No policy: open the scene in MuJoCo's interactive viewer. The Control sliders in its right-hand
        # panel drive the arm (motor1-6: joint targets in radians; either finger slider: the gripper).
        # Double-click a body, then ctrl + right-drag to push it; ] and [ look through the cameras.
        env.reset(arm_start, objects=objects)
        model, data = env.model, env.data
        arm = [model.actuator(f"motor{i + 1}").id for i in range(6)]
        fingers = [model.actuator(f"finger_{joint}").id for joint in scene.FINGER_JOINTS]
        # Give the sliders the joints' own travel (without a range the viewer offers only -1..1).
        model.actuator_ctrllimited[arm + fingers] = 1
        model.actuator_ctrlrange[arm] = [model.joint(name).range for name in scene.ARM_JOINTS]
        model.actuator_ctrlrange[fingers] = [0.0, scene.FINGER_TRAVEL]
        target = np.asarray(arm_start, dtype=float).copy()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Show the wrist camera's photographed room (real 3D surfaces); hide the overhead camera's flat sheets.
            for geom in range(model.ngeom):
                if model.geom(geom).name.startswith("overhead_"):
                    model.geom_group[geom] = scene.GROUP_COLLISION
            viewer.opt.geomgroup[scene.GROUP_OVERHEAD_ONLY] = 1
            viewer.opt.geomgroup[scene.GROUP_WRIST_ONLY] = 1
            viewer.opt.geomgroup[scene.GROUP_COLLISION] = 0
            while viewer.is_running():
                start = time.time()
                with viewer.lock():
                    # Whatever the sliders now say is the new command. The two finger sliders are one
                    # gripper: follow whichever was moved.
                    target[:6] = data.ctrl[arm]
                    sent = target[6] * scene.FINGER_TRAVEL
                    moved = max(data.ctrl[fingers], key=lambda value: abs(value - sent))
                    target[6] = moved / scene.FINGER_TRAVEL
                    env.advance(target)
                viewer.sync()
                time.sleep(max(0.0, 1.0 / env.config.control_hz - (time.time() - start)))
        env.close()
        return

    policy_args = PolicyArgs(policy_path=args.policy_path, max_steps=args.max_steps, device=args.device)
    policy = LeRobotPolicy(policy_args.policy_path, policy_args.task, policy_args.device)
    success, steps = rollout(env, policy, arm_start, setup, policy_args, video=args.video)
    print(f"ball on {target}: {success} after {steps} steps ({steps / 30:.1f} s); video: {args.video}")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
