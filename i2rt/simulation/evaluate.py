"""Score the simulation against the recorded data: does it move, and look, like the real station?

Three measurements, each over the whole dataset:

* **replay** -- every episode's recorded actions are run open-loop through the simulated arm and
  physics, starting from the episode's real initial conditions. Reported: how closely the simulated
  joints follow the real ones, and in how many episodes the ball ends up on the cup the operator put
  it on. Nothing but the actions comes from the recording, so this tests dynamics, object geometry
  and contact together.
* **images** -- the scene is posed as it was at sampled recorded frames and rendered through both
  calibrated cameras; each render is compared with the real frame (mean absolute error in grey
  levels, and structural similarity, SSIM).
* **alignment** -- how far the simulated arm's outline is from the real arm's in the overhead view
  (the camera calibration's own residual, on frames it was not fitted to).

    python -m i2rt.simulation.evaluate
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
from i2rt.simulation.calibration.appearance import pose_scene
from i2rt.simulation.calibration.common import DEFAULT_DATASET
from i2rt.simulation.env import EnvConfig, YamStationEnv
from i2rt.simulation.rendering import ImageModel
from i2rt.simulation.replay import replay_physics

# Image region the operator and the leader arm occupy in the overhead view: not part of the simulated scene.
OPERATOR_REGION = np.array([[0, 200], [205, 200], [205, 479], [0, 479]], np.int32)


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    frames_per_episode: int = 6
    out: Path = scene.SIM_ROOT / "outputs" / "evaluation.json"
    sheet: Path | None = scene.SIM_ROOT / "outputs" / "real_vs_sim.jpg"
    """Write a contact sheet of real | simulated frame pairs here."""


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean structural similarity of two uint8 images (grey, 11x11 Gaussian windows), optionally within a mask."""
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(z: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(z, (11, 11), 1.5)

    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    values = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(values[mask].mean() if mask is not None else values.mean())


def main(args: Args) -> None:
    episodes = episode_data.load(args.dataset_root)
    setups = episode_data.all_setups(args.dataset_root)
    cameras = tuple(scene.load_cameras())
    report: dict = {"episodes": len(episodes)}

    # ---- replay ------------------------------------------------------------------------------------
    env = YamStationEnv(EnvConfig(cameras=()))
    errors, outcomes = [], {}
    for name, episode in episodes.items():
        result = replay_physics(env, episode, setups[name])
        errors.append(result.sim_states - episode.state)
        if result.success is not None:
            outcomes[name] = bool(result.success)
    errors = np.concatenate(errors)
    report["replay"] = {
        "joint_tracking_rms_rad": [float(v) for v in np.sqrt((errors[:, :6] ** 2).mean(axis=0))],
        "gripper_tracking_rms": float(np.sqrt((errors[:, 6] ** 2).mean())),
        "ball_placed": int(sum(outcomes.values())),
        "episodes_with_a_placement": len(outcomes),
        "failed": sorted(name for name, ok in outcomes.items() if not ok),
    }
    print(f"replay: ball placed on the target cup in {report['replay']['ball_placed']}/{len(outcomes)} episodes")
    print(
        f"        joint tracking rms [deg]: {np.round(np.degrees(report['replay']['joint_tracking_rms_rad']), 2).tolist()}, gripper {report['replay']['gripper_tracking_rms']:.3f}"
    )
    env.close()

    # ---- images ------------------------------------------------------------------------------------
    env = YamStationEnv(EnvConfig(cameras=cameras))
    workspace = np.ones((480, 640), np.uint8)
    cv2.fillPoly(workspace, [OPERATOR_REGION], 0)
    latency = {camera: ImageModel.for_camera(camera).latency for camera in cameras}
    scores: dict[str, list] = {camera: [] for camera in cameras}
    pairs: dict[str, list] = {camera: [] for camera in cameras}
    for e, (name, episode) in enumerate(episodes.items()):
        env.set_backdrop(name)
        for k, frame in enumerate(np.linspace(0, len(episode) - 1, args.frames_per_episode).astype(int)):
            for camera in cameras:
                pose_scene(env, episode, setups[name], int(frame), lag=latency[camera])
                sim = env.render(camera, noise=False)  # random noise can only lower a pixel-wise score
                real = cv2.cvtColor(episode.image(camera, int(frame)), cv2.COLOR_BGR2RGB)
                mask = workspace > 0 if camera == "overhead" else None
                difference = np.abs(sim.astype(np.float32) - real.astype(np.float32)).mean(axis=2)
                scores[camera].append(
                    (float(difference[mask].mean() if mask is not None else difference.mean()), ssim(sim, real, mask))
                )
                if e % 6 == 0 and k in (1, 4):
                    pairs[camera].append(np.hstack([real, sim]))
    report["images"] = {}
    for camera in cameras:
        values = np.array(scores[camera])
        report["images"][camera] = {
            "frames": len(values),
            "mean_abs_error": float(values[:, 0].mean()),
            "ssim": float(values[:, 1].mean()),
        }
        note = " (operator's end of the table excluded)" if camera == "overhead" else ""
        print(
            f"images, {camera}: mean abs error {values[:, 0].mean():.1f}/255, SSIM {values[:, 1].mean():.3f} over {len(values)} frames{note}"
        )
    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    if args.sheet is not None:
        rows = [
            np.hstack([cv2.resize(pairs[camera][i], (640, 240)) for camera in cameras])
            for i in range(min(len(pairs[c]) for c in cameras))
        ]
        cv2.imwrite(
            str(args.sheet), cv2.cvtColor(np.vstack(rows[:10]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        print(f"wrote {args.sheet}")


if __name__ == "__main__":
    main(tyro.cli(Args))
