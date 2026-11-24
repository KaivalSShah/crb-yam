"""Fit where the room's light comes from, by the shadows on the table.

The table in the overhead view is a photograph, and it is drawn lit so that simulated objects can
shade it. Which way their shadows fall, how dark they are and how soft is set by a cluster of
directional lights. Here those are chosen so that the table *around* the arm, the cups and the box
-- where their shadows are -- looks in the simulation as it does in the recorded frames.

    python -m i2rt.simulation.calibration.lighting
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro
from scipy.optimize import minimize

from i2rt.simulation import episodes as episode_data
from i2rt.simulation import scene
from i2rt.simulation.calibration.appearance import pose_scene
from i2rt.simulation.calibration.common import CALIB_DIR, DEFAULT_DATASET, select_static_frames
from i2rt.simulation.env import EnvConfig, YamStationEnv

NEAR_PX = 45  # how far from an object its shadow is looked for
FITTED = ("lean_deg", "heading_deg", "spread_deg", "shadow_strength", "shadow_blur")
# The lower limit on the shadow blur is a rendering one: the fit would rather have crisp shadows, but
# MuJoCo's shadow maps are blocky, and a cluster of them left unblurred reads as speckle, not shadow.
BOUNDS = {
    "lean_deg": (0.0, 60.0),
    "heading_deg": (-720.0, 720.0),
    "spread_deg": (2.0, 35.0),
    "shadow_strength": (0.0, 1.0),
    "shadow_blur": (5.0, 25.0),
}


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    frames: int = 40
    out: Path = CALIB_DIR / "lighting.json"


def main(args: Args) -> None:
    episodes = episode_data.load(args.dataset_root)
    names = list(episodes)
    setups = episode_data.all_setups(args.dataset_root)
    env = YamStationEnv(EnvConfig(cameras=("overhead",)))
    env.render("overhead")
    renderer = env._renderers["overhead"]
    model = env.model
    table = model.geom("overhead_table").id
    static = {table, model.geom("overhead_backdrop").id}
    lighting = scene.lighting_parameters()

    # Per frame: the posed scene, the real picture, and which table pixels lie next to something that casts a shadow.
    samples = []
    for e, f in select_static_frames(list(episodes.values()), args.frames, seed=3):
        episode = episodes[names[e]]
        pose_scene(env, episode, setups[names[e]], f)
        seg = renderer.render_segmentation(env.data)
        casters = (seg >= 0) & ~np.isin(seg, list(static))
        near = cv2.dilate(casters.astype(np.uint8), np.ones((2 * NEAR_PX + 1, 2 * NEAR_PX + 1), np.uint8)) > 0
        around = near & (seg == table) & ~(cv2.dilate(casters.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        around[:, :205] = False  # the operator's end of the table
        real = cv2.cvtColor(episode.image("overhead", f), cv2.COLOR_BGR2RGB).astype(np.float32).mean(axis=2)
        samples.append((names[e], env.data.qpos.copy(), real, around))

    def error(values: np.ndarray) -> float:
        trial = {**lighting, **dict(zip(FITTED, values, strict=True))}
        if any(not BOUNDS[k][0] <= trial[k] <= BOUNDS[k][1] for k in FITTED):
            return 1e3
        scene.apply_lighting(model, trial)
        renderer.shadow_strength, renderer.shadow_blur = trial["shadow_strength"], trial["shadow_blur"]
        total = 0.0
        for name, qpos, real, around in samples:
            env.set_backdrop(name)
            env.data.qpos[:] = qpos
            env.holding_ball = False
            scene.mujoco.mj_forward(model, env.data)
            sim = renderer.render(env.data, effects=False).astype(np.float32).mean(axis=2)
            total += float(np.abs(sim - real)[around].mean())
        return total / len(samples)

    start = np.array([lighting[k] for k in FITTED], dtype=float)
    print(f"table next to objects, mean abs error before: {error(start):.2f} grey levels", flush=True)
    # Which way the light leans has several local optima (a shadow on the wrong side is as bad from
    # either side), so look all the way round before refining.
    best = (error(start), start)
    for lean in (15.0, 30.0, 45.0):
        for heading in np.arange(0.0, 360.0, 30.0):
            trial = start.copy()
            trial[0], trial[1] = lean, heading
            score = error(trial)
            if score < best[0]:
                best = (score, trial)
    print(f"coarse search: {best[0]:.2f} at lean {best[1][0]:.0f} deg, heading {best[1][1]:.0f} deg", flush=True)
    result = minimize(
        error,
        best[1],
        method="Nelder-Mead",
        options={
            "maxiter": 160,
            "xatol": 0.05,
            "fatol": 0.002,
            "initial_simplex": np.vstack([best[1], best[1] + np.diag([6.0, 15.0, 5.0, 0.1, 3.0])]),
        },
    )
    fitted = {**lighting, **{k: float(v) for k, v in zip(FITTED, result.x, strict=True)}}
    fitted["heading_deg"] %= 360.0
    fitted["notes"] = (
        "light cluster fitted to the shadows on the table in the overhead footage (calibration/lighting.py)"
    )
    args.out.write_text(json.dumps(fitted, indent=2) + "\n")
    print(f"after: {result.fun:.2f} grey levels  ->  " + ", ".join(f"{k} {fitted[k]:.1f}" for k in FITTED))
    print(f"saved {args.out}")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
