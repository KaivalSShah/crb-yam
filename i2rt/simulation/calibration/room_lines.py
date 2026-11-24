"""Measure the overhead camera's vertical focal length from the room's own straight lines.

Fitting a camera to the arm's silhouette (``overhead_camera.py``) leaves one direction poorly
determined: a longer lens further away draws the arm almost the same. The room settles it. Its
vertical edges (conduits, door and window frames) all point at one vanishing point in the image,
and the horizontal edges of the partition beside the table at another. Vertical and horizontal are
perpendicular, which for a camera whose principal point is the image centre gives the focal length
directly:

    fy^2 = (v_vertical.y - cy) * (cy - v_horizontal.y)        (both vanishing points lie near x = cx)

For the recorded footage this gives fy = 508 px, i.e. a 16:9 sensor with an 88 degree diagonal field
of view squashed to 640x480 -- which is what the camera's data sheet says.

    python -m i2rt.simulation.calibration.room_lines
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro

from i2rt.simulation.calibration.common import DEFAULT_DATASET, load_episodes, modal_background


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    min_length: float = 40.0
    """Shortest line segment [px] used."""


def vanishing_point(segments: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Least-squares vanishing point of image segments (N, 4), with outliers rejected.

    Returns (point in homogeneous pixels, median angular residual in degrees, segments kept).
    """
    p0 = np.c_[segments[:, :2], np.ones(len(segments))]
    p1 = np.c_[segments[:, 2:], np.ones(len(segments))]
    lines = np.cross(p0, p1)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    length = np.hypot(*(segments[:, 2:] - segments[:, :2]).T)
    middle = (segments[:, :2] + segments[:, 2:]) / 2.0
    direction = (segments[:, 2:] - segments[:, :2]) / length[:, None]
    keep = np.ones(len(segments), bool)
    error = np.zeros(len(segments))
    point = np.zeros(3)
    for _ in range(4):
        point = np.linalg.svd(lines[keep] * length[keep, None])[2][-1]
        to_point = np.c_[point[0] - middle[:, 0] * point[2], point[1] - middle[:, 1] * point[2]]
        to_point /= np.linalg.norm(to_point, axis=1, keepdims=True)
        error = np.degrees(np.arcsin(np.abs(to_point[:, 0] * direction[:, 1] - to_point[:, 1] * direction[:, 0])))
        keep = error < max(1.0, 2.0 * np.median(error))
    return point / point[2], float(np.median(error[keep])), int(keep.sum())


def estimate(dataset_root: Path, min_length: float = 40.0) -> dict:
    """Vanishing points of the room's vertical and horizontal edges and the focal length they imply."""
    plate, _ = modal_background(load_episodes(dataset_root), "overhead")
    grey = cv2.cvtColor(np.clip(plate, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    segments = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV).detect(grey)[0].reshape(-1, 4)
    length = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    angle = np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0])) % 180
    x, y = (segments[:, 0] + segments[:, 2]) / 2.0, (segments[:, 1] + segments[:, 3]) / 2.0
    long_enough = length > min_length
    # Where each family of edges is in this camera's view: vertical edges on the back wall and the
    # far frames (upper right), horizontal edges along the partition beside the table (right).
    vertical = vanishing_point(segments[long_enough & (angle > 84) & (angle < 110) & (y < 150) & (x > 330)])
    horizontal = vanishing_point(segments[long_enough & (angle > 58) & (angle < 80) & (x > 430)])
    cy = (grey.shape[0] - 1) / 2.0
    fy = float(np.sqrt((vertical[0][1] - cy) * (cy - horizontal[0][1])))
    return {"fy": fy, "vertical": vertical, "horizontal": horizontal, "image_height": grey.shape[0]}


def main(args: Args) -> None:
    result = estimate(args.dataset_root, args.min_length)
    for label in ("vertical", "horizontal"):
        point, error, count = result[label]
        print(
            f"{label + ' edges:':17s} {count} segments, vanishing point ({point[0]:.0f}, {point[1]:.0f}), residual {error:.2f} deg"
        )
    fy = result["fy"]
    print(f"fy = {fy:.0f} px   (pass to overhead_camera.py as --fy {fy:.0f}; with a 16:9 sensor, --aspect 0.75)")
    diagonal = 2 * np.degrees(np.arctan(np.hypot(1920, 1080) / 2 / (fy * 1080 / result["image_height"])))
    print(f"as a 1920x1080 sensor that is a {diagonal:.0f} degree diagonal field of view")


if __name__ == "__main__":
    main(tyro.cli(Args))
