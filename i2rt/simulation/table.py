"""The table top as a plane in the robot's base frame.

The simulation's world frame is the arm's base frame, because that is the frame its joint angles,
and therefore every calibration here, refer to. The table the arm is bolted to is *nearly* the plane
z = 0 of that frame, but not exactly: the mount gives a little under the arm's weight and the joint
zeros are only so accurate, and at arm's length a degree or two is centimetres. The grasps in the
recorded data (the fingers closing on a ball that stands on a cup that stands on the table) measure
where the table really is as the arm sees it: z = a + b*x + c*y.

Everything that rests on the table uses this plane, and gravity points along its normal, so the
physics is that of a level table with the arm mounted slightly off-square to it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parent / "calibration" / "results" / "table_plane.json"


@dataclass(frozen=True)
class TablePlane:
    a: float = 0.0  # height at the base origin [m]
    b: float = 0.0  # slope along +X (the arm's reach direction)
    c: float = 0.0  # slope along +Y

    @classmethod
    def load(cls, path: Path = _PATH) -> TablePlane:
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text())
        return cls(float(data["a"]), float(data["b"]), float(data["c"]))

    def save(self, path: Path = _PATH, notes: str = "") -> None:
        path.write_text(json.dumps({**asdict(self), "notes": notes}, indent=2) + "\n")

    @property
    def normal(self) -> np.ndarray:
        n = np.array([-self.b, -self.c, 1.0])
        return n / np.linalg.norm(n)

    @property
    def quat(self) -> np.ndarray:
        """Rotation (w, x, y, z) taking +Z onto the plane's normal by the shortest arc."""
        n = self.normal
        axis = np.cross([0.0, 0.0, 1.0], n)
        sin_half = np.linalg.norm(axis)
        if sin_half < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        angle = np.arctan2(sin_half, n[2])
        return np.concatenate([[np.cos(angle / 2.0)], axis / sin_half * np.sin(angle / 2.0)])

    def height(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return self.a + self.b * x + self.c * y

    def point(self, xy: np.ndarray, lift: float = 0.0) -> np.ndarray:
        """3D point ``lift`` metres off the table (along its normal) above table position ``xy``."""
        xy = np.asarray(xy, dtype=float)
        return np.array([xy[0], xy[1], self.height(xy[0], xy[1])]) + lift * self.normal

    def intersect(self, origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """Where rays from ``origin`` along ``directions`` (N, 3) meet the table."""
        directions = np.atleast_2d(directions)
        numerator = self.a + self.b * origin[0] + self.c * origin[1] - origin[2]
        denominator = directions[:, 2] - self.b * directions[:, 0] - self.c * directions[:, 1]
        return origin + (numerator / denominator)[:, None] * directions

    def lift_of(self, point: np.ndarray) -> float:
        """Signed distance of ``point`` from the table, along its normal."""
        return float((np.asarray(point) - np.array([0.0, 0.0, self.a])) @ self.normal)
