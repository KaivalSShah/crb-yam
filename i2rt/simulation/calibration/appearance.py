"""Match the simulation's paints to what each real camera recorded.

For a set of recorded frames the scene is posed as it was (arm at the logged joints, cups where the
overhead camera saw them) and rendered through a calibrated camera together with a map of which
material each pixel shows. Wherever a material is visible, away from its edges, the simulated and
the real pixel are two readings of the same surface. Each material's colour is then scaled so the
medians agree. Run per camera, because each camera has its own colour response.

    python -m i2rt.simulation.calibration.appearance
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
import tyro

from i2rt.simulation import episodes as episode_data
from i2rt.simulation import scene
from i2rt.simulation.calibration.common import CALIB_DIR, DEFAULT_DATASET, Episode, select_static_frames
from i2rt.simulation.calibration.objects import box_mask as overhead_box_mask
from i2rt.simulation.calibration.objects import colour_mask as overhead_mask
from i2rt.simulation.calibration.wrist_camera import real_masks as wrist_masks
from i2rt.simulation.env import EnvConfig, YamStationEnv

# Materials whose paint is fitted, per camera (the wrist camera never sees the arm's links).
FITTED = {
    "overhead": (
        "robot_white",
        "robot_black",
        "robot_finger",
        "camera_bracket",
        "cup_orange_plastic",
        "cup_blue_plastic",
        "ball_plastic",
        "cardboard",
    ),
    "right_wrist": ("robot_finger", "cup_orange_plastic", "cup_blue_plastic", "ball_plastic", "cardboard"),
}


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    frames: int = 90
    passes: int = 3
    out: Path = CALIB_DIR / "appearance.json"


def pose_scene(
    env: YamStationEnv, episode: Episode, setup: episode_data.EpisodeSetup, frame: int, lag: float = 0.0
) -> None:
    """Pose the scene as it was at a recorded frame, without physics.

    The arm goes to the logged joints and the objects to where the episode had them: the ball on its
    cup before the grasp and after the release, and in the gripper in between (at the spot the ball
    typically sits in the grasp, ``ball_in_gripper_at_grasp`` from ``objects.py``).

    ``lag`` is the latency of the camera the pose is for: the frame saved with joint reading ``frame``
    shows the arm where it was ``lag`` seconds earlier, so that is where the arm is put.
    """
    state = episode.state[frame]
    if lag > 0:
        when = episode.timestamp[frame] - lag
        state = np.array(
            [np.interp(when, episode.timestamp, episode.state[:, j]) for j in range(episode.state.shape[1])]
        )
    env.reset(state, objects=setup.objects)
    env.holding_ball = setup.grasp_frame is not None and setup.grasp_frame <= frame <= setup.release_frame
    adr = env.model.joint("ball_free").qposadr[0]
    params = scene.object_parameters()
    resting = scene.ball_rest_height(params) + params["ball"]["radius"]
    if setup.grasp_frame is None:
        env.data.qpos[adr : adr + 3] = [3.0, -3.0, 0.5]
    elif setup.grasp_frame - 3 <= frame <= setup.release_frame + 2:
        gripper = env.data.body("gripper")
        in_hand = np.array(scene.load_json("objects.json").get("ball_in_gripper_at_grasp", [0.0, 0.0, 0.13]))
        env.data.qpos[adr : adr + 3] = gripper.xpos + gripper.xmat.reshape(3, 3) @ in_hand
    else:
        cup = setup.source_cup if frame < setup.grasp_frame else setup.target_cup
        env.data.qpos[adr : adr + 3] = env.table.point(np.array(setup.objects[cup].xy), resting)
    mujoco.mj_forward(env.model, env.data)


def real_object_masks(camera: str, image: np.ndarray) -> dict[str, np.ndarray]:
    """Where the real frame itself shows each coloured object (by colour), keyed by material name.

    Simulated and real objects do not overlap perfectly, least of all close to the wrist camera, and
    a material's colour must not be fitted to whatever happens to lie under its simulated outline.
    """
    if camera == "right_wrist":
        masks = wrist_masks(image)
        return {
            "cup_orange_plastic": masks["cup_orange"],
            "cup_blue_plastic": masks["cup_blue"],
            "ball_plastic": masks["ball"],
            "robot_finger": masks["tips"],
            "cardboard": masks["cardboard"],
        }
    return {
        **{f"{name}_plastic": overhead_mask(image, name) for name in ("cup_orange", "cup_blue", "ball")},
        "cardboard": overhead_box_mask(image),
    }


def main(args: Args) -> None:
    episodes = episode_data.load(args.dataset_root)
    names = list(episodes)
    setups = episode_data.all_setups(args.dataset_root)
    chosen = select_static_frames(list(episodes.values()), args.frames)
    result = json.loads(args.out.read_text()) if args.out.is_file() else {}
    for camera in (c for c in FITTED if c in scene.load_cameras()):
        for sweep in range(args.passes):
            env = YamStationEnv(EnvConfig(cameras=(camera,)))
            env.render(camera)  # builds the renderer
            renderer = env._renderers[camera]
            model = env.model
            material_of_geom = np.array([model.material(m).name if m >= 0 else "" for m in model.geom_matid] + [""])
            samples: dict[str, list] = {name: [] for name in FITTED[camera]}
            for e, f in chosen:
                episode = episodes[names[e]]
                pose_scene(env, episode, setups[names[e]], f)
                sim = renderer.render(env.data, noise=False).astype(np.float32)
                seg = renderer.render_segmentation(env.data)
                real = cv2.cvtColor(episode.image(camera, f), cv2.COLOR_BGR2RGB).astype(np.float32)
                shown = material_of_geom[seg]
                confirmed = real_object_masks(camera, episode.image(camera, f))
                for name in FITTED[camera]:
                    inner = cv2.erode((shown == name).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                    if name in confirmed:  # only where the real frame shows that object too
                        inner &= confirmed[name]
                    if inner.sum() >= 20:
                        samples[name].append((np.median(sim[inner], axis=0), np.median(real[inner], axis=0)))
            colours = result.setdefault(camera, {})
            report = []
            for name, pairs in samples.items():
                if len(pairs) < 3:
                    continue
                sim_rgb = np.median([p[0] for p in pairs], axis=0)
                real_rgb = np.median([p[1] for p in pairs], axis=0)
                current = np.array(colours.get(name, model.mat_rgba[model.material(name).id].tolist()))
                # Saturated channels carry no ratio information: leave those as they are.
                gain = np.where((sim_rgb > 4) & (sim_rgb < 252), (real_rgb + 1.0) / (sim_rgb + 1.0), 1.0)
                updated = np.clip(current[:3] * np.clip(gain, 0.5, 2.0), 0.005, 1.6)
                colours[name] = [*(float(v) for v in updated), 1.0]
                report.append(f"{name}: sim {sim_rgb.round(0).tolist()} real {real_rgb.round(0).tolist()}")
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            env.close()
            print(f"[{camera}] pass {sweep + 1}\n  " + "\n  ".join(report), flush=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
