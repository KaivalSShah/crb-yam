"""Run a trained LeRobot policy (ACT, pi0.5, ...) on the simulated station.

The simulation speaks the dataset's own format, so a policy trained on the recorded demonstrations
is given exactly what it was trained on -- ``observation.state``, ``observation.images.overhead``,
``observation.images.right_wrist`` and the task string -- and its actions drive the simulated arm
the way they would drive the real one. Each rollout starts from the initial conditions of a recorded
episode (arm pose, cup, ball and box positions).

    python -m i2rt.simulation.run_policy --policy-path outputs/train/yam_act/checkpoints/last/pretrained_model

Requires LeRobot (``uv pip install lerobot``). A rollout counts as a success when the ball comes to
rest on the cup it did not start on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from i2rt.simulation import episodes as episode_data
from i2rt.simulation import scene
from i2rt.simulation.env import IMAGE_KEY, STATE_KEY, EnvConfig, YamStationEnv
from i2rt.simulation.replay import ball_on_cup


@dataclass
class Args:
    policy_path: str
    """Checkpoint folder (the ``pretrained_model`` directory ``lerobot-train`` writes) or a Hub repo id."""
    task: str = "place the ball on top of the other cup"
    episodes: int = 10
    max_steps: int = 450
    """Control steps per rollout (30 Hz: 450 is 15 s; the demonstrations take about 9 s)."""
    device: str = "auto"
    """``cuda``, ``mps`` or ``cpu``; ``auto`` picks the first available."""
    max_joint_speed: float = 2.0
    """Limit on how fast a joint target may move [rad/s], as the teleop loop applies to the real arm."""
    jitter: float = 0.0
    """Move the cups by up to this many metres from where the recorded episode had them (0: exact recorded starts)."""
    out: Path = scene.SIM_ROOT / "outputs" / "policy_rollouts"
    seed: int = 0


class LeRobotPolicy:
    """A LeRobot checkpoint as a plain ``observation -> action`` function."""

    def __init__(self, path: str, task: str, device: str) -> None:
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        except ImportError as error:
            raise SystemExit(f"LeRobot is not installed ({error}). Run: uv pip install lerobot") from error
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        config = PreTrainedConfig.from_pretrained(path)
        config.device = device
        self.policy = get_policy_class(config.type).from_pretrained(path, config=config)
        self.policy.to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            config, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}}
        )
        self.task = task

    def reset(self) -> None:
        self.policy.reset()

    def __call__(self, observation: dict) -> np.ndarray:
        torch = self._torch
        batch: dict = {"task": [self.task]}
        for key, value in observation.items():
            if key == STATE_KEY:
                batch[key] = torch.from_numpy(np.asarray(value, dtype=np.float32))[None]
            else:  # HWC uint8 -> 1CHW float in [0, 1], as LeRobot datasets deliver images
                batch[key] = torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)[None].float() / 255.0
        with torch.inference_mode():
            action = self.postprocess(self.policy.select_action(self.preprocess(batch)))
        return action[0].to("cpu").numpy().astype(np.float64)


def rollout(
    env: YamStationEnv,
    policy: Callable[[dict], np.ndarray],
    start_state: np.ndarray,
    setup: episode_data.EpisodeSetup,
    args: Args,
    video: Path | None,
) -> tuple[bool, int]:
    """One rollout from a starting condition. Returns (success, steps taken)."""
    observation = env.reset(start_state, objects=setup.objects)
    for _ in range(15):
        env.advance(start_state)
    observation = env.observe()
    if hasattr(policy, "reset"):
        policy.reset()
    cameras = [c for c in env.config.cameras if IMAGE_KEY.format(c) in observation]
    target = setup.target_cup
    step_limit = args.max_joint_speed / env.config.control_hz
    command = observation[STATE_KEY].astype(np.float64)
    container = stream = None
    if video is not None:
        import av  # PyAV: comes with LeRobot

        container = av.open(str(video), "w")
        stream = container.add_stream("libx264", rate=round(env.config.control_hz))
        stream.width, stream.height, stream.pix_fmt = 640 * len(cameras), 480, "yuv420p"
    success_steps, steps = 0, 0
    while steps < args.max_steps:
        steps += 1
        action = np.asarray(policy(observation), dtype=np.float64)
        command[:6] += np.clip(action[:6] - command[:6], -step_limit, step_limit)
        command[6] = np.clip(action[6], 0.0, 1.0)
        observation = env.step(command)
        if container is not None:
            frame = np.hstack([observation[IMAGE_KEY.format(c)] for c in cameras])
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                container.mux(packet)
        placed = target is not None and ball_on_cup(env, target)[0] and not env.holding_ball
        success_steps = success_steps + 1 if placed else 0
        if success_steps >= round(env.config.control_hz):  # on the cup, released, for a full second
            break
    if container is not None:
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    return success_steps >= round(env.config.control_hz), steps


def main(args: Args) -> None:
    policy = LeRobotPolicy(args.policy_path, args.task, args.device)
    env = YamStationEnv(EnvConfig(cameras=tuple(scene.load_cameras())))
    args.out.mkdir(parents=True, exist_ok=True)
    wins = 0
    for i in range(args.episodes):
        name, state, setup = episode_data.sample_start(np.random.default_rng([args.seed, i]), args.jitter)
        env.set_backdrop(name)
        success, steps = rollout(env, policy, state, setup, args, args.out / f"rollout_{i:03d}_{name}.mp4")
        wins += success
        print(
            f"rollout {i}: start of {name}, {'SUCCESS' if success else 'failed'} after {steps} steps ({steps / env.config.control_hz:.1f} s)",
            flush=True,
        )
    print(f"success rate: {wins}/{args.episodes}; videos in {args.out}")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
