"""Record simulated episodes in the same raw format the real teleop recorder writes.

Each recorded real episode is replayed in the simulation (its actions through the simulated motors,
objects starting where they really were) and saved as an episode folder with the real recorder's
layout -- ``overhead/`` and ``right_wrist/`` JPEGs, ``state.npy``, ``action.npy``,
``timestamp.npy``, ``meta.json`` -- so ``i2rt.robot_learning.convert_to_lerobot`` and everything
downstream read simulated data exactly like real data.

    python -m i2rt.simulation.record --out data/ball_pick_and_place_sim
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro

from i2rt.simulation import episodes as episode_data
from i2rt.simulation import scene
from i2rt.simulation.calibration.common import DEFAULT_DATASET
from i2rt.simulation.env import IMAGE_KEY, STATE_KEY, EnvConfig, YamStationEnv
from i2rt.simulation.replay import ball_on_cup

JPEG_QUALITY = 95  # as i2rt.robot_learning.episode_recorder


@dataclass
class Args:
    out: Path
    """Folder to write ``episode_NNNN`` folders into."""
    dataset_root: Path = DEFAULT_DATASET
    """Real recordings whose actions and starting conditions are replayed."""
    episodes: tuple[str, ...] = ()
    """Episode names to replay (default: all)."""
    only_successful: bool = False
    """Keep an episode only if the simulated ball ends up on the target cup."""


def write_episode(
    folder: Path,
    frames: dict[str, list[np.ndarray]],
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    meta: dict,
) -> None:
    for camera, images in frames.items():
        (folder / camera).mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            cv2.imwrite(
                str(folder / camera / f"{i:06d}.jpg"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
    np.save(folder / "state.npy", states.astype(np.float32))
    np.save(folder / "action.npy", actions.astype(np.float32))
    np.save(folder / "timestamp.npy", timestamps)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))  # last: marks the episode complete


def main(args: Args) -> None:
    episodes = episode_data.load(args.dataset_root)
    cameras = tuple(scene.load_cameras())
    env = YamStationEnv(EnvConfig(cameras=cameras))
    args.out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for name in args.episodes or tuple(episodes):
        episode = episodes[name]
        setup = episode_data.episode_setup(episode)
        env.set_backdrop(name)
        obs = env.reset(episode.state[0], objects=setup.objects)
        for _ in range(15):  # let the objects settle before the clock starts
            env.advance(episode.action[0])
        obs = env.observe()
        states, frames = [obs[STATE_KEY]], {c: [obs[IMAGE_KEY.format(c)]] for c in cameras}
        for k in range(1, len(episode)):
            obs = env.step(episode.action[k], duration=float(episode.timestamp[k] - episode.timestamp[k - 1]))
            states.append(obs[STATE_KEY])
            for camera in cameras:
                frames[camera].append(obs[IMAGE_KEY.format(camera)])
        success = None if setup.target_cup is None else bool(ball_on_cup(env, setup.target_cup)[0])
        if args.only_successful and not success:
            print(f"{name}: ball did not end on {setup.target_cup}; skipped", flush=True)
            continue
        meta = {**episode.meta, "frames": len(episode), "simulated_from": name, "sim_success": success}
        write_episode(args.out / name, frames, np.array(states), episode.action, episode.timestamp, meta)
        kept += 1
        print(f"{name}: {len(episode)} frames, ball on target cup: {success}", flush=True)
    env.close()
    print(f"wrote {kept} simulated episodes to {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
