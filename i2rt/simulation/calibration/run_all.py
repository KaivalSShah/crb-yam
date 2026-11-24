"""Calibrate the whole simulation from a recorded dataset, in dependency order.

    python -m i2rt.simulation.calibration.run_all                       # everything (about an hour)
    python -m i2rt.simulation.calibration.run_all --steps objects bake_overhead appearance

Each step is one of the scripts in this folder; its docstring says what it measures and how. Run on
a new dataset (a moved camera, a different table) this rebuilds every measured number and texture
the simulation uses. The one thing it does not decide is the table plane
(``results/table_plane.json``): see the README for how that was chosen and how to re-check it.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro

from i2rt.simulation.calibration import room_lines
from i2rt.simulation.calibration.common import DEFAULT_DATASET
from i2rt.simulation.table import TablePlane

STEPS = (
    "overhead_camera",  # pose + focal length of the fixed camera, from the arm's silhouette and the room's lines
    "objects",  # cup and box sizes; where they stood in every episode
    "wrist_camera",  # the camera on the gripper, from the fingertips, the cups and the ball
    "objects_again",  # cups hidden from above, now that the wrist camera can look for them
    "grasp_events",  # report: do the gripper's grasps agree with where the cameras put the cups?
    "latency",  # how stale each camera's frames are relative to the joint readings
    "bake_overhead",  # the photographed table and room the overhead camera sees
    "dynamics",  # joint friction, damping, inertia and gravity terms, from command/response pairs
    "lighting",  # where the light comes from, by the shadows on the table
    "bake_wrist",  # the photographed room the wrist camera sees
    "sheen",  # the table's view-dependent gloss in the wrist camera
    "appearance",  # paints, per camera
    "evaluate",  # how close the result is to the recordings
)


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    steps: tuple[str, ...] = STEPS
    aspect: float = 0.75
    """fx/fy of both cameras: 0.75 for 16:9 sensors recorded at 640x480."""


def main(args: Args) -> None:
    unknown = [step for step in args.steps if step not in STEPS]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}; choose from {list(STEPS)}")
    data = ["--dataset-root", str(args.dataset_root)]
    plane = TablePlane.load()
    for step in args.steps:
        module, extra = f"i2rt.simulation.calibration.{step}", list(data)
        if step == "overhead_camera":
            fy = room_lines.estimate(args.dataset_root)["fy"]
            extra += ["--fy", f"{fy:.1f}", "--aspect", str(args.aspect), "--no-distortion", "--frames", "100"]
        elif step == "wrist_camera":
            extra += [
                "--aspect",
                str(args.aspect),
                "--no-fit-plane",
                "--plane",
                str(plane.a),
                str(plane.b),
                str(plane.c),
                "--scales",
                "0.5",
            ]
        elif step == "objects_again":
            module = "i2rt.simulation.calibration.objects"
        elif step == "latency":
            extra += ["--write"]
        elif step == "evaluate":
            module = "i2rt.simulation.evaluate"
        start = time.time()
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, "-m", module, *extra], check=True)
        print(f"=== {step} done in {time.time() - start:.0f}s ===", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
