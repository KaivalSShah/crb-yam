"""Pinhole camera model shared by calibration, rendering and object localisation.

Conventions
-----------
* ``pos``/``quat`` are the camera pose in the frame of the body it is attached to (the world for
  the overhead camera, the gripper body for the wrist camera), in MuJoCo's camera convention:
  the camera looks along its local ``-Z`` with ``+X`` right and ``+Y`` up.
* ``fx, fy, cx, cy`` are in pixels of the *recorded* image (640x480), in OpenCV's convention:
  integer coordinates are pixel centres. The recorder squashes each camera's native frame to
  640x480 without keeping its aspect ratio, so ``fx != fy`` in general.
* ``k1`` is a single radial distortion coefficient in the OpenCV convention, applied about the
  principal point in normalised coordinates. MuJoCo renders an ideal pinhole image; ``distort``
  warps such a render into the distorted image the real lens produces.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np

from i2rt.simulation.table import TablePlane

# MuJoCo camera axes (x right, y up, z backward) -> OpenCV camera axes (x right, y down, z forward).
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


@dataclass
class PinholeCamera:
    name: str
    pos: list[float]
    quat: list[float]  # w, x, y, z (MuJoCo camera convention)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480
    k1: float = 0.0
    k2: float = 0.0
    parent_body: str = "world"
    notes: dict = field(default_factory=dict)

    # ---- persistence -------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> PinholeCamera:
        return cls(**json.loads(path.read_text()))

    # ---- geometry ----------------------------------------------------------------------------
    @property
    def rotation(self) -> np.ndarray:
        """3x3 rotation, camera (MuJoCo convention) -> parent."""
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, np.asarray(self.quat, dtype=float))
        return mat.reshape(3, 3)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array([self.k1, self.k2, 0.0, 0.0, 0.0])

    def world_to_cv(self, parent_pos: np.ndarray | None = None, parent_mat: np.ndarray | None = None) -> tuple:
        """(R, t) with x_cv = R @ x_world + t, given the parent body's world pose (identity if None)."""
        r_cam = self.rotation
        p_cam = np.asarray(self.pos, dtype=float)
        if parent_mat is not None:
            r_cam = parent_mat @ r_cam
            p_cam = parent_mat @ p_cam + (parent_pos if parent_pos is not None else 0.0)
        elif parent_pos is not None:
            p_cam = p_cam + parent_pos
        r = (r_cam @ _MJ_TO_CV).T
        return r, -r @ p_cam

    def project(
        self,
        points_world: np.ndarray,
        parent_pos: np.ndarray | None = None,
        parent_mat: np.ndarray | None = None,
        distort: bool = True,
    ) -> np.ndarray:
        """Project Nx3 world points to Nx2 pixels (and return depth as a third column)."""
        r, t = self.world_to_cv(parent_pos, parent_mat)
        pc = np.atleast_2d(points_world) @ r.T + t
        z = pc[:, 2:3]
        xn = pc[:, :2] / z
        if distort and (self.k1 or self.k2):
            r2 = (xn**2).sum(axis=1, keepdims=True)
            xn = xn * (1.0 + self.k1 * r2 + self.k2 * r2**2)
        uv = xn * np.array([self.fx, self.fy]) + np.array([self.cx, self.cy])
        return np.hstack([uv, z])

    def pixel_ray(
        self, uv: np.ndarray, parent_pos: np.ndarray | None = None, parent_mat: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """World-frame ray (origin, unit directions Nx3) through each pixel, undoing lens distortion."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float)).reshape(-1, 1, 2)
        xn = cv2.undistortPoints(uv, self.K, self.dist_coeffs).reshape(-1, 2)
        d_cv = np.hstack([xn, np.ones((len(xn), 1))])
        r, t = self.world_to_cv(parent_pos, parent_mat)
        d_world = d_cv @ r  # rows: R^T @ d
        d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
        origin = -r.T @ t
        return origin, d_world

    def pixel_to_plane(self, uv: np.ndarray, plane_z: float = 0.0) -> np.ndarray:
        """Intersect pixel rays with the horizontal world plane ``z = plane_z`` (world-fixed cameras)."""
        origin, d = self.pixel_ray(uv)
        s = (plane_z - origin[2]) / d[:, 2]
        return origin + s[:, None] * d

    def pixel_to_table(self, uv: np.ndarray, table: TablePlane) -> np.ndarray:
        """Where the rays through pixels ``uv`` meet the table (world-fixed cameras)."""
        origin, d = self.pixel_ray(uv)
        return table.intersect(origin, d)

    # ---- MuJoCo / image-space helpers ------------------------------------------------------------
    def apply_to_model(self, model: mujoco.MjModel, cam_id: int, canvas_scale: float = 1.0) -> None:
        """Write pose + intrinsics into an existing MuJoCo camera (declared with ``sensorsize``).

        MuJoCo renders an ideal pinhole image. ``canvas_scale`` > 1 keeps the focal length in pixels
        but widens the rendered canvas about its centre, which gives ``distortion_maps`` the margin
        a barrel-distorting lens needs at the image corners. The render target must keep the
        camera's 4:3 shape; its pixel count is free (supersampling).
        """
        model.cam_pos[cam_id] = self.pos
        model.cam_quat[cam_id] = self.quat
        sw, sh = model.cam_sensorsize[cam_id]
        canvas_w, canvas_h = self.width * canvas_scale, self.height * canvas_scale
        # MuJoCo stores intrinsics in sensor length units, the principal point as an offset from the
        # image centre whose sign is (centre - c) on both axes (checked against rendered points).
        # It measures from the image's corner; (cx, cy) follow OpenCV and index pixel centres,
        # hence the half pixel.
        model.cam_intrinsic[cam_id] = [
            self.fx * sw / canvas_w,
            self.fy * sh / canvas_h,
            (self.width / 2.0 - (self.cx + 0.5)) * sw / canvas_w,
            (self.height / 2.0 - (self.cy + 0.5)) * sh / canvas_h,
        ]

    def distortion_maps(
        self, render_w: int, render_h: int, canvas_scale: float = 1.0, out_scale: float = 1.0
    ) -> tuple:
        """``cv2.remap`` maps taking a pinhole render (see ``apply_to_model``) to the real, distorted image.

        The output has ``out_scale`` times the camera's resolution; each of its pixels looks up where
        the lens actually imaged it from in the ideal render of ``render_w`` x ``render_h`` pixels.
        """
        out_w, out_h = round(self.width * out_scale), round(self.height * out_scale)
        u, v = np.meshgrid(
            (np.arange(out_w, dtype=np.float64) + 0.5) / out_scale - 0.5,
            (np.arange(out_h, dtype=np.float64) + 0.5) / out_scale - 0.5,
        )
        if self.k1 or self.k2:
            pts = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
            xn = cv2.undistortPoints(pts, self.K, self.dist_coeffs).reshape(out_h, out_w, 2)
            u, v = xn[..., 0] * self.fx + self.cx, xn[..., 1] * self.fy + self.cy
        canvas_w, canvas_h = self.width * canvas_scale, self.height * canvas_scale
        # Pixel centres: final pixel u sits at canvas coordinate u + margin, then scale to the render.
        map_x = (u + 0.5 + (canvas_w - self.width) / 2.0) * (render_w / canvas_w) - 0.5
        map_y = (v + 0.5 + (canvas_h - self.height) / 2.0) * (render_h / canvas_h) - 0.5
        return map_x.astype(np.float32), map_y.astype(np.float32)


def look_at_quat(pos: np.ndarray, target: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """MuJoCo-convention camera quaternion at ``pos`` looking at ``target`` with world +Z up, rolled by ``roll``."""
    fwd = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c, s = np.cos(roll), np.sin(roll)
    right, up = c * right + s * up, -s * right + c * up
    mat = np.stack([right, up, -fwd], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat
