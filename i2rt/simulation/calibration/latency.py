"""Measure how stale each camera's frames are relative to the joint states logged with them.

The recorder saves, at each tick, the latest joint reading and the latest frame of each camera. The
joint reading is fresh; the frame was exposed some time earlier (exposure, USB transfer, decoding).
While the arm moves that difference is visible: the frame shows the arm where it *was*.

So, for the frames recorded while the arm moved fastest, the scene is rendered at the joint state
of a little earlier or later (interpolated between readings), and the lag at which the rendered
outlines best fit the real frames is the camera's latency. The overhead camera is scored on the
arm's outline, the wrist camera on the cups' (the fingertips ride with the camera and say nothing).

    python -m i2rt.simulation.calibration.latency

Recorded footage: about 80 ms for the overhead camera and 55 ms for the wrist camera (whose minimum
is broad: 50 to 80 ms score within a few percent of each other). The values go
in ``results/image_models.json`` (``latency``); the environment then hands out frames that old.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from i2rt.simulation.calibration import wrist_camera
from i2rt.simulation.calibration.common import CALIB_DIR, DEFAULT_DATASET, Episode, load_episodes, modal_background
from i2rt.simulation.calibration.overhead_camera import SilhouetteScorer, real_dark_mask, table_prediction
from i2rt.simulation.cameras import PinholeCamera
from i2rt.simulation.table import TablePlane

OFFSETS = np.arange(-0.14, 0.061, 0.02)  # seconds; negative: the frame shows the arm where it was earlier


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    per_episode: int = 4
    """Fastest frames taken from each episode."""
    write: bool = False
    """Store the result in ``results/image_models.json``."""


def state_at(episode: Episode, frame: int, offset: float) -> np.ndarray:
    when = episode.timestamp[frame] + offset
    return np.array([np.interp(when, episode.timestamp, episode.state[:, j]) for j in range(episode.state.shape[1])])


def fastest_frames(episodes: list[Episode], per_episode: int, minimum: float = 0.4) -> list[tuple[Episode, int]]:
    picks = []
    for episode in episodes:
        s, t = episode.state, episode.timestamp
        speed = np.abs(s[2:, :6] - s[:-2, :6]).max(axis=1) / (t[2:] - t[:-2])
        for index in np.argsort(-speed)[: per_episode * 3]:
            frame = int(index) + 1
            if 6 <= frame < len(episode) - 6 and speed[frame - 1] > minimum:
                picks.append((episode, frame))
    return picks


def best_lag(scores: list[float]) -> float:
    """Lag [s] at the minimum of a parabola through the scores around their smallest."""
    i = int(np.clip(np.argmin(scores), 1, len(scores) - 2))
    a, b, _ = np.polyfit(OFFSETS[i - 1 : i + 2], scores[i - 1 : i + 2], 2)
    return float(b / (2 * a))  # minus the vertex: a positive lag means the frame is older


def main(args: Args) -> None:
    episodes = load_episodes(args.dataset_root)
    result = {}

    # ---- overhead: the arm's own outline ---------------------------------------------------------------
    camera = PinholeCamera.load(CALIB_DIR / "overhead_camera.json")
    background, support = modal_background(episodes, "overhead")
    table_grey = table_prediction(background, support)
    picks = fastest_frames(episodes, args.per_episode)[: args.per_episode * len(episodes)]
    masks = [real_dark_mask(episode.image("overhead", frame), table_grey) for episode, frame in picks]
    scores = []
    for offset in OFFSETS:
        frames = [
            (state_at(episode, frame, offset), mask) for (episode, frame), mask in zip(picks, masks, strict=True)
        ]
        scores.append(SilhouetteScorer(frames, scale=0.5).score(camera))
    result["overhead"] = best_lag(scores)
    print(
        f"overhead: {len(picks)} fast frames; outline mismatch by lag: "
        + ", ".join(f"{-o * 1e3:.0f} ms {s:.2f}" for o, s in zip(OFFSETS, scores, strict=True))
    )
    print(f"overhead camera latency: {result['overhead'] * 1e3:.0f} ms")

    # ---- wrist: the cups, as the camera sweeps past them ----------------------------------------------
    camera = PinholeCamera.load(CALIB_DIR / "wrist_camera.json")
    overhead = PinholeCamera.load(CALIB_DIR / "overhead_camera.json")
    objects = json.loads((CALIB_DIR / "objects.json").read_text())
    table = TablePlane.load()
    samples = []
    for episode, frame in fastest_frames(episodes, args.per_episode * 3):
        seen = wrist_camera.real_masks(episode.image("right_wrist", frame))
        if seen["cup_orange"].sum() + seen["cup_blue"].sum() < 2500:
            continue
        rays = {}
        for name, entry in objects["episodes"][episode.name].items():
            if entry and name != "box":
                origin, direction = overhead.pixel_ray(np.array([entry["base_px"]]))
                rays[name] = (origin.tolist(), direction[0].tolist())
        samples.append((episode, frame, {"cup_rays": rays, "masks": seen, "ball_on": None}))
    samples = samples[: args.per_episode * len(episodes)]
    scores = []
    for offset in OFFSETS:
        posed = [{**sample, "state": state_at(episode, frame, offset)} for episode, frame, sample in samples]
        scores.append(wrist_camera.WristScorer(posed, scale=0.5).score(camera, table))
    result["right_wrist"] = best_lag(scores)
    print(
        f"wrist: {len(samples)} fast frames; outline mismatch by lag: "
        + ", ".join(f"{-o * 1e3:.0f} ms {s:.2f}" for o, s in zip(OFFSETS, scores, strict=True))
    )
    print(f"wrist camera latency: {result['right_wrist'] * 1e3:.0f} ms")

    if args.write:
        path = CALIB_DIR / "image_models.json"
        models = json.loads(path.read_text()) if path.is_file() else {}
        for name, lag in result.items():
            models.setdefault(name, {})["latency"] = round(max(lag, 0.0), 3)
        path.write_text(json.dumps(models, indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
