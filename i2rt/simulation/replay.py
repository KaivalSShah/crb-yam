"""Replay a recorded episode in the simulation and compare it with the real footage.

Two modes:

* ``physics`` -- the scene starts as the episode did (arm pose, cups, ball) and is then driven only
  by the recorded *actions*, through the simulated motors and contact physics. Nothing else from the
  recording is used, so whether the ball ends up on the other cup is a test of the whole replica.
* ``kinematic`` -- the arm is placed at the recorded joint *states* frame by frame and the ball is
  carried along while the gripper was closed. No physics: this isolates what the cameras see.

Either writes a side-by-side video (real | simulated, per camera) and prints how far apart the two are.

    python -m i2rt.simulation.replay --episode episode_0020
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tyro

from i2rt.simulation import episodes as episode_data
from i2rt.simulation import scene
from i2rt.simulation.calibration.appearance import pose_scene
from i2rt.simulation.calibration.common import DEFAULT_DATASET, Episode
from i2rt.simulation.env import IMAGE_KEY, EnvConfig, YamStationEnv
from i2rt.simulation.rendering import ImageModel


@dataclass
class Args:
    episode: str = "episode_0020"
    mode: Literal["physics", "kinematic"] = "physics"
    dataset_root: Path = DEFAULT_DATASET
    out: Path | None = None
    """Video path. Defaults to ``outputs/<episode>_<mode>.mp4`` inside the simulation folder."""
    cameras: tuple[str, ...] = ("overhead", "right_wrist")
    every: int = 1
    """Render every n-th frame (the physics always runs on all of them)."""


@dataclass
class ReplayResult:
    sim_states: np.ndarray  # (T, 7) simulated joint states at the recorded frame times
    ball: np.ndarray  # (T, 3) simulated ball centre
    frames: dict[str, list[np.ndarray]]  # camera -> rendered RGB frames (possibly subsampled)
    frame_indices: list[int]
    success: bool | None  # ball resting on the target cup at the end (None if the episode has no target)
    ball_to_target: float | None  # horizontal distance [m] from the ball to the target cup at the end


def ball_on_cup(env: YamStationEnv, cup: str) -> tuple[bool, float]:
    """Whether the ball is sitting on ``cup``: over its base, at resting height, and not held."""
    params = scene.object_parameters()
    offset = env.object_position("ball") - env.object_position(cup)
    height = float(offset @ env.table.normal)
    horizontal = float(np.linalg.norm(offset - height * env.table.normal))
    resting = scene.ball_rest_height(params) + params["ball"]["radius"]
    return horizontal < params["cup"]["top_radius"] and abs(height - resting) < 0.012, horizontal


def replay_physics(
    env: YamStationEnv,
    episode: Episode,
    setup: episode_data.EpisodeSetup,
    cameras: tuple[str, ...] = (),
    every: int = 1,
) -> ReplayResult:
    env.reset(episode.state[0], objects=setup.objects)
    # Let the objects settle under the arm's starting pose before the clock starts.
    for _ in range(15):
        env.advance(episode.action[0])
    states, ball, frames, indices = [], [], {c: [] for c in cameras}, []
    for k in range(len(episode)):
        observation = None
        if k > 0:
            duration = float(episode.timestamp[k] - episode.timestamp[k - 1])
            if cameras:  # the frames a camera would deliver with this joint reading: its latency included
                observation = env.step(episode.action[k], duration=duration)
            else:
                env.advance(episode.action[k], duration=duration)
        states.append(env.joint_state())
        ball.append(env.object_position("ball"))
        if cameras and k % every == 0:
            indices.append(k)
            for camera in cameras:
                frames[camera].append(observation[IMAGE_KEY.format(camera)] if observation else env.render(camera))
    for _ in range(30):  # one more second holding the last command, so a dropped ball comes to rest
        env.advance(episode.action[-1])
    success, distance = (None, None) if setup.target_cup is None else ball_on_cup(env, setup.target_cup)
    return ReplayResult(np.array(states), np.array(ball), frames, indices, success, distance)


def replay_kinematic(
    env: YamStationEnv, episode: Episode, setup: episode_data.EpisodeSetup, cameras: tuple[str, ...], every: int = 1
) -> ReplayResult:
    """Pose the scene at every recorded frame (no physics) and render it, each camera with its own latency."""
    latency = {camera: ImageModel.for_camera(camera).latency for camera in cameras}
    states, ball, frames, indices = [], [], {c: [] for c in cameras}, []
    for k in range(0, len(episode), every):
        indices.append(k)
        for camera in cameras:
            pose_scene(env, episode, setup, k, lag=latency[camera])
            frames[camera].append(env.render(camera))
        pose_scene(env, episode, setup, k)
        states.append(env.joint_state())
        ball.append(env.object_position("ball"))
    return ReplayResult(np.array(states), np.array(ball), frames, indices, None, None)


def write_comparison_video(path: Path, episode: Episode, result: ReplayResult, fps: float = 30.0) -> None:
    """Side-by-side video: one row per camera, real on the left, simulation on the right."""
    try:
        import av
    except ImportError as error:
        raise SystemExit(f"writing videos needs PyAV ({error}). Run: uv pip install av") from error
    cameras = [c for c in result.frames if result.frames[c]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=round(fps))
        stream.width, stream.height = 1280, 480 * len(cameras)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for i, k in enumerate(result.frame_indices):
            rows = []
            for camera in cameras:
                real = cv2.cvtColor(episode.image(camera, k), cv2.COLOR_BGR2RGB)
                sim = result.frames[camera][i].copy()
                cv2.putText(
                    real, f"real  {camera}  frame {k}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )
                cv2.putText(sim, "simulation", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                rows.append(np.hstack([real, sim]))
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.vstack(rows), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main(args: Args) -> None:
    episodes = episode_data.load(args.dataset_root)
    if args.episode not in episodes:
        raise SystemExit(
            f"{args.episode} not found under {args.dataset_root} (have {next(iter(episodes))} .. {list(episodes)[-1]})"
        )
    episode = episodes[args.episode]
    setup = episode_data.episode_setup(episode)
    cameras = tuple(c for c in args.cameras if c in scene.load_cameras())
    env = YamStationEnv(EnvConfig(cameras=cameras))
    env.set_backdrop(args.episode)
    run = replay_physics if args.mode == "physics" else replay_kinematic
    result = run(env, episode, setup, cameras, args.every)
    reference = episode.state if len(result.sim_states) == len(episode) else episode.state[result.frame_indices]
    error = result.sim_states - reference
    print(f"{args.episode} ({len(episode)} frames, {args.mode}): ball {setup.source_cup} -> {setup.target_cup}")
    rms = [round(float(v), 4) for v in np.sqrt((error**2).mean(axis=0))]
    print(f"  joint tracking error vs the real arm, rms [rad x6, gripper]: {rms}")
    if result.success is not None:
        print(
            f"  ball on the target cup at the end: {result.success} ({result.ball_to_target * 1e3:.0f} mm from its centre)"
        )
    out = args.out or scene.SIM_ROOT / "outputs" / f"{args.episode}_{args.mode}.mp4"
    write_comparison_video(out, episode, result, fps=30.0 / args.every)
    print(f"  wrote {out}")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
