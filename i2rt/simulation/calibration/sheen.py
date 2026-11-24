"""Measure the table's sheen as the wrist camera sees it.

The table top is glossy: on top of its flat grey it mirrors a blurred image of the ceiling lights,
and that highlight slides across the table as the camera moves. A texture cannot do that -- it is
the same from every angle -- so the wrist camera's photographed table (``bake_wrist.py``) only
carries the *average* brightness of each spot.

What the highlight depends on is the direction a pixel's line of sight leaves in after bouncing off
the table. So for every table pixel of every sampled wrist frame, that mirror direction is computed
from the logged arm pose, and the difference between the real pixel and the photographed table is
filed under it. Averaged over the dataset this is a small map -- mirror direction -> how much
brighter or darker than average the table looks -- which is in effect a blurred picture of the
ceiling, as reflected by this table. ``rendering.py`` adds it back to table pixels at render time.

    python -m i2rt.simulation.calibration.sheen
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro

from i2rt.simulation import episodes as episode_data
from i2rt.simulation.calibration.appearance import pose_scene
from i2rt.simulation.calibration.common import DEFAULT_DATASET, SIM_ROOT
from i2rt.simulation.calibration.wrist_camera import real_masks
from i2rt.simulation.env import EnvConfig, YamStationEnv

MAP_SIZE = 48  # bins per axis over the horizontal components of the mirror direction, each in [-1, 1]


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    every: int = 8
    out: Path = SIM_ROOT / "assets" / "wrist_table_sheen.npy"


def table_axes(table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: ANN001
    """(u, v, normal): two unit vectors spanning the table and its normal, the frame the map is indexed in."""
    normal = table.normal
    u = np.cross([0.0, 1.0, 0.0], normal)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u), normal


def mirror_coordinates(rays: np.ndarray, table) -> np.ndarray:  # noqa: ANN001
    """Map coordinates in [0, MAP_SIZE) of the directions ``rays`` (N, 3, unit) leave in after reflecting off the table."""
    u, v, normal = table_axes(table)
    reflected = rays - 2.0 * (rays @ normal)[:, None] * normal
    return np.stack(
        [(reflected @ u + 1.0) * 0.5 * (MAP_SIZE - 1), (reflected @ v + 1.0) * 0.5 * (MAP_SIZE - 1)], axis=1
    )


def main(args: Args) -> None:
    episodes = episode_data.load(args.dataset_root)
    setups = episode_data.all_setups(args.dataset_root)
    env = YamStationEnv(EnvConfig(cameras=("right_wrist",)))
    env.render("right_wrist")
    renderer = env._renderers["right_wrist"]
    renderer.sheen = None  # measure against the plain photographed table
    camera, model, data = renderer.camera, env.model, env.data
    table_geom = model.geom("wrist_table").id
    gripper = model.body(camera.parent_body).id
    u, v = np.meshgrid(np.arange(camera.width, dtype=np.float64), np.arange(camera.height, dtype=np.float64))
    normalised = cv2.undistortPoints(
        np.stack([u, v], axis=-1).reshape(-1, 1, 2), camera.K, camera.dist_coeffs
    ).reshape(-1, 2)
    rays_cv = np.hstack([normalised, np.ones((len(normalised), 1))])
    rays_cv /= np.linalg.norm(rays_cv, axis=1, keepdims=True)

    sums = np.zeros((MAP_SIZE * MAP_SIZE, 3))
    counts = np.zeros(MAP_SIZE * MAP_SIZE)
    for name, episode in episodes.items():
        for frame in range(0, len(episode), args.every):
            pose_scene(env, episode, setups[name], frame, lag=renderer.image_model.latency)
            sim = renderer.render(data, effects=False).astype(np.float32)
            shows_table = renderer.render_segmentation(data) == table_geom
            real = cv2.cvtColor(episode.image("right_wrist", frame), cv2.COLOR_BGR2RGB).astype(np.float32)
            masks = real_masks(cv2.cvtColor(real.astype(np.uint8), cv2.COLOR_RGB2BGR))
            other = masks["ball"] | masks["cup_orange"] | masks["cup_blue"] | masks["tips"] | masks["cardboard"]
            keep = (
                shows_table & ~cv2.dilate(other.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            ).reshape(-1)
            keep &= (
                np.abs(real.reshape(-1, 3) - sim.reshape(-1, 3)).max(axis=1) < 110
            )  # not table after all (a hand, the box)
            rotation, _ = camera.world_to_cv(data.xpos[gripper], data.xmat[gripper].reshape(3, 3))
            rays = rays_cv[keep] @ rotation
            coords = np.round(mirror_coordinates(rays, env.table)).astype(int)
            index = coords[:, 1] * MAP_SIZE + coords[:, 0]
            residual = (real.reshape(-1, 3) - sim.reshape(-1, 3))[keep]
            for channel in range(3):
                sums[:, channel] += np.bincount(index, weights=residual[:, channel], minlength=len(counts))
            counts += np.bincount(index, minlength=len(counts))
    seen = (counts > 200).reshape(MAP_SIZE, MAP_SIZE)
    sheen = (sums / np.maximum(counts, 1)[:, None]).reshape(MAP_SIZE, MAP_SIZE, 3).astype(np.float32)
    # Directions never observed: carry the nearest observed value outward, then smooth the whole map
    # (the highlight is a blur of the ceiling to begin with).
    weight = seen.astype(np.float32)
    filled = sheen * weight[..., None]
    for sigma in (1.5, 4.0, 10.0):
        denominator = cv2.GaussianBlur(weight, (0, 0), sigma)
        estimate = cv2.GaussianBlur(sheen * seen[..., None], (0, 0), sigma) / np.maximum(denominator, 1e-6)[..., None]
        take = (weight == 0) & (denominator > 0.02)
        filled[take] = estimate[take]
        weight = np.where(take, 1.0, weight)
    sheen = cv2.GaussianBlur(filled, (0, 0), 1.0)
    np.save(args.out, sheen)
    grey = sheen.mean(axis=2)
    print(
        f"mirror directions observed: {seen.mean() * 100:.0f}% of the map; sheen ranges from {grey[seen].min():+.0f} to {grey[seen].max():+.0f} grey levels around the average"
    )
    print(f"saved {args.out}")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
