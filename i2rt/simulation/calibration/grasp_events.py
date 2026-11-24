"""Collect the moments the gripper closed on, and let go of, the ball: a table-plane ground truth.

In every episode the ball is picked off one cup and set down on the other. While the fingers hold
it, the ball -- and so the cup under it -- lies on the plane midway between the two fingers. The
arm's forward kinematics give that plane; the overhead camera, independently, gives the cup's
position on the table. Along the fingers' closing direction the two must agree.

That is a constraint the arm's silhouette alone does not provide: the arm is seen 10-40 cm above
the table, so a camera fitted to it can trade focal length against height and still be centimetres
off for points *on* the table. ``overhead_camera.py --grasps`` uses these events to remove that
freedom. The cup is recorded as a pixel (the image of its base centre), so the events stay valid
whichever camera model is being tried.

    python -m i2rt.simulation.calibration.grasp_events
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro

from i2rt.simulation import scene
from i2rt.simulation.calibration.common import CALIB_DIR, DEFAULT_DATASET, load_episodes, set_robot_state
from i2rt.simulation.cameras import PinholeCamera
from i2rt.simulation.table import TablePlane

GRIPPER_CLOSING = 0.75


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    camera: Path = CALIB_DIR / "overhead_camera.json"
    """The camera ``objects.json`` was produced with (used only to turn cup positions back into pixels)."""
    objects: Path = CALIB_DIR / "objects.json"
    out: Path = CALIB_DIR / "grasp_events.json"


def finger_frame(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """(point midway between the fingertips, unit closing direction), both in the world frame."""
    closing = data.xaxis[model.joint("joint7").id]  # the finger's slide axis
    return data.site("grasp_site").xpos.copy(), closing / np.linalg.norm(closing)


def main(args: Args) -> None:
    camera = PinholeCamera.load(args.camera)
    table = TablePlane.load()  # the plane ``objects.json`` was produced with
    objects = json.loads(args.objects.read_text())
    model = scene.build_gravity_model()
    data = mujoco.MjData(model)
    events: dict[str, list] = {
        "episode": [],
        "kind": [],
        "frame": [],
        "cup": [],
        "site_xy": [],
        "site_z": [],
        "closing_xy": [],
        "cup_base_px": [],
    }
    for episode in load_episodes(args.dataset_root):
        cups = {k: np.array(v["xy"]) for k, v in objects["episodes"].get(episode.name, {}).items() if v}
        closed = np.flatnonzero(episode.action[:, 6] < GRIPPER_CLOSING)
        if len(closed) < 5 or len(cups) < 2:
            continue
        for kind, frame in (("grasp", int(closed[0]) + 6), ("release", int(closed[-1]) - 2)):
            set_robot_state(model, data, episode.state[frame])
            site, closing = finger_frame(model, data)
            horizontal = closing[:2] / max(np.linalg.norm(closing[:2]), 1e-9)
            if np.linalg.norm(closing[:2]) < 0.8:  # fingers closing mostly vertically say little about xy
                continue
            cup = min(cups, key=lambda name: np.linalg.norm(cups[name] - site[:2]))
            if np.linalg.norm(cups[cup] - site[:2]) > 0.06:
                continue  # not at a cup: an aborted grasp, or a cup that was not found in the image
            events["episode"].append(episode.name)
            events["kind"].append(kind)
            events["frame"].append(frame)
            events["cup"].append(cup)
            events["site_xy"].append(site[:2].tolist())
            events["site_z"].append(float(site[2]))
            events["closing_xy"].append(horizontal.tolist())
            events["cup_base_px"].append(camera.project(table.point(cups[cup])[None])[0, :2].tolist())
    args.out.write_text(json.dumps(events, indent=1) + "\n")
    on_table = camera.pixel_to_table(np.array(events["cup_base_px"]), table)[:, :2]
    offset = on_table - np.array(events["site_xy"])
    along = 1e3 * np.einsum("ij,ij->i", offset, np.array(events["closing_xy"]))
    print(f"{len(along)} events -> {args.out}")
    print(
        f"cup off the fingers' mid-plane under the current camera: mean {along.mean():+.1f} mm, rms {np.sqrt((along**2).mean()):.1f} mm"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
