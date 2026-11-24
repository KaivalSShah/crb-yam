"""Fit the overhead camera (pose + intrinsics) to the recorded footage, using the arm as the target.

The arm's joint angles are logged with every frame, so its black wrist and gripper are a moving
calibration object whose 3D pose is known. For a candidate camera we render the arm's silhouette at
the logged joint angles and score it against the dark pixels of the matching real frame:

* every rendered wrist/gripper pixel should land on a real dark pixel, and
* every real dark pixel on the table should be explained by some part of the rendered arm.

Both are capped chamfer distances, summed over a set of still frames spread over the workspace.
The fit runs a global search at low resolution, then a local refinement at full resolution.

    python -m i2rt.simulation.calibration.overhead_camera
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
import tyro
from scipy.optimize import differential_evolution, minimize

from i2rt.simulation.calibration.common import (
    CALIB_DIR,
    DEFAULT_DATASET,
    load_episodes,
    modal_background,
    robot_model_with_camera,
    select_static_frames,
    set_robot_state,
)
from i2rt.simulation.cameras import PinholeCamera, look_at_quat

# Image region used for scoring: the tabletop in front of the arm. Left of it is the operator and
# the leader arm, right of it the cardboard box and power brick, above it the floor and wall.
SCORE_POLYGON = np.array([[205, 200], [432, 200], [432, 330], [455, 470], [205, 470]], dtype=np.int32)
DISTAL_BODIES = ("link4", "link5", "gripper", "tip_left", "tip_right")  # black on the real arm
CAP_PX = 40.0


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    frames: int = 60
    """Still frames used for the fit."""
    global_frames: int = 14
    """Subset used by the low-resolution global search."""
    init: Path | None = None
    """Start the refinement from this camera instead of running the global search."""
    aspect: float | None = None
    """Pin fx/fy (0.75 for a 16:9 sensor squashed to 4:3). Free when omitted."""
    fy: float | None = None
    """Pin the vertical focal length [px], e.g. from ``room_lines.py``'s vanishing points. Free when omitted."""
    distortion: bool = True
    """Fit radial lens distortion (k1)."""
    k2: bool = False
    """Also fit the second radial coefficient."""
    scales: tuple[float, ...] = (0.5, 1.0)
    """Image scales the refinement runs at, in order."""
    grasps: Path | None = None
    """``grasp_events.json`` (from ``grasp_events.py``): adds the table-plane constraint described there."""
    grasp_weight: float = 0.5
    """Pixels of silhouette error one millimetre of grasp inconsistency is worth."""
    iters: int = 700
    seed: int = 0
    out: Path = CALIB_DIR / "overhead_camera.json"
    debug_dir: Path | None = None
    """Write overlay images of the fit here."""


def table_prediction(background: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Grey level the bare table would have at each pixel, interpolated across the arm's ghost."""
    grey = background.mean(axis=2)
    region = np.zeros(grey.shape, np.uint8)
    cv2.fillPoly(region, [np.array([[0, 200], [470, 200], [545, 480], [0, 480]], np.int32)], 1)
    weight = ((support > 0.75) & (region > 0) & (grey > 95)).astype(np.float32)
    num = cv2.GaussianBlur(grey * weight, (0, 0), 22)
    den = cv2.GaussianBlur(weight, (0, 0), 22)
    return num / np.maximum(den, 1e-3)


def real_dark_mask(image: np.ndarray, table_grey: np.ndarray) -> np.ndarray:
    """Pixels much darker than the bare table: the arm's black parts (plus a little shadow/cable)."""
    grey = image.astype(np.float32).mean(axis=2)
    dark = (grey < 0.58 * table_grey).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # drops the thin USB cable
    return dark > 0


class SilhouetteScorer:
    """Renders arm silhouettes for a candidate camera and scores them against the real masks."""

    CANVAS = 1.3  # pinhole render margin, so a barrel-distorting lens has pixels to pull in at the corners

    def __init__(self, frames: list[tuple[np.ndarray, np.ndarray]], scale: float = 1.0) -> None:
        self.scale = scale
        self.w, self.h = round(640 * scale), round(480 * scale)
        self.render_w, self.render_h = round(self.w * self.CANVAS), round(self.h * self.CANVAS)
        self.model = robot_model_with_camera()
        self.cam_id = self.model.camera("calib").id
        self.renderer = mujoco.Renderer(self.model, self.render_h, self.render_w)
        self.renderer.enable_segmentation_rendering()
        distal_ids = {self.model.body(b).id for b in DISTAL_BODIES}
        self.geom_is_distal = np.array([b in distal_ids for b in self.model.geom_bodyid])
        region = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(region, [SCORE_POLYGON], 1)
        self.region = cv2.resize(region, (self.w, self.h), interpolation=cv2.INTER_NEAREST) > 0
        self.cap = CAP_PX * scale
        self.datas, self.real, self.real_dt = [], [], []
        for state, mask in frames:
            data = mujoco.MjData(self.model)
            set_robot_state(self.model, data, state)
            self.datas.append(data)
            small = cv2.resize(mask.astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST) > 0
            small &= self.region
            self.real.append(small)
            dt = cv2.distanceTransform((~small).astype(np.uint8), cv2.DIST_L2, 3)
            self.real_dt.append(np.minimum(dt, self.cap))
        self._maps_key, self._maps = None, None
        self.grasps: dict | None = None
        self.grasp_weight = 0.0

    def use_grasps(self, grasps: dict, weight: float) -> None:
        """Also require cups seen on the table to sit where the gripper closed on the ball (see grasp_events.py)."""
        self.grasps = {k: np.asarray(v, dtype=float) for k, v in grasps.items()}
        self.grasp_weight = weight

    def grasp_error_mm(self, camera: PinholeCamera) -> np.ndarray:
        """Per grasp/release: how far the cup, located through ``camera``, sits off the fingers' mid-plane."""
        cup_xy = camera.pixel_to_plane(self.grasps["cup_base_px"])[:, :2]
        offset = cup_xy - self.grasps["site_xy"]
        return 1e3 * np.einsum("ij,ij->i", offset, self.grasps["closing_xy"])

    def _lens_maps(self, camera: PinholeCamera) -> tuple[np.ndarray, np.ndarray]:
        key = (camera.fx, camera.fy, camera.cx, camera.cy, camera.k1, camera.k2)
        if key != self._maps_key:
            self._maps = camera.distortion_maps(self.render_w, self.render_h, self.CANVAS, out_scale=self.scale)
            self._maps_key = key
        return self._maps

    def render_classes(self, camera: PinholeCamera, index: int) -> tuple[np.ndarray, np.ndarray]:
        """(any arm pixel, distal black pixel) masks for frame ``index``, as the real lens images them."""
        camera.apply_to_model(self.model, self.cam_id, canvas_scale=self.CANVAS)
        data = self.datas[index]
        mujoco.mj_camlight(self.model, data)  # the renderer reads the camera pose from data, not the model
        self.renderer.update_scene(data, camera=self.cam_id)
        seg = self.renderer.render()
        geom = seg[..., 0]
        robot = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom >= 0)
        distal = robot & self.geom_is_distal[np.clip(geom, 0, len(self.geom_is_distal) - 1)]
        labels = robot.astype(np.uint8) + distal.astype(np.uint8)
        map_x, map_y = self._lens_maps(camera)
        labels = cv2.remap(labels, map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
        return labels >= 1, labels == 2

    def score(self, camera: PinholeCamera) -> float:
        total = 0.0
        for i in range(len(self.datas)):
            robot, distal = self.render_classes(camera, i)
            distal &= self.region
            real = self.real[i]
            term1 = self.real_dt[i][distal].mean() if distal.any() else self.cap
            if real.any():
                if robot.any():
                    dt = cv2.distanceTransform((~robot).astype(np.uint8), cv2.DIST_L2, 3)
                    term2 = np.minimum(dt[real], self.cap).mean()
                else:
                    term2 = self.cap
            else:
                term2 = 0.0
            total += (term1 + term2) / (2.0 * self.scale)
        score = total / len(self.datas)
        if self.grasps is not None:
            score += self.grasp_weight * float(np.minimum(np.abs(self.grasp_error_mm(camera)), 40.0).mean())
        return score


def camera_from_lookat(p: np.ndarray, name: str = "overhead") -> PinholeCamera:
    """Global-search parametrisation: position, look-at point on the table, roll, fy, fx/fy."""
    pos = p[0:3]
    quat = look_at_quat(pos, np.array([p[3], p[4], 0.0]), roll=p[5])
    return PinholeCamera(name, list(pos), list(quat), fx=p[6] * p[7], fy=p[6], cx=319.5, cy=239.5)


# Refinement parameters: translation (3), rotation vector (3), log fy, log(fx/fy), cx, cy, k1, k2.
REFINE_STEP = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 8.0, 8.0, 0.08, 0.05])


def camera_from_local(base: PinholeCamera, p: np.ndarray, aspect: float | None = None) -> PinholeCamera:
    """Refinement parametrisation: a small change about ``base``. ``aspect`` pins fx/fy when given."""
    angle = float(np.linalg.norm(p[3:6]))
    dq = np.array([1.0, 0.0, 0.0, 0.0])
    if angle > 1e-12:
        mujoco.mju_axisAngle2Quat(dq, p[3:6] / angle, angle)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.asarray(base.quat, dtype=float), dq)
    fy = base.fy * np.exp(p[6])
    fx = aspect * fy if aspect is not None else base.fx * np.exp(p[6] + p[7])
    return PinholeCamera(
        base.name,
        [float(v) for v in np.asarray(base.pos) + p[0:3]],
        [float(v) for v in quat],
        fx=float(fx),
        fy=float(fy),
        cx=float(base.cx + p[8]),
        cy=float(base.cy + p[9]),
        k1=float(base.k1 + p[10]),
        k2=float(base.k2 + p[11]),
    )


def refine(
    camera: PinholeCamera, scorer: SilhouetteScorer, free: np.ndarray, aspect: float | None, iters: int
) -> tuple:
    """Nelder-Mead over the ``free`` subset of the refinement parameters, restarted once."""
    idx = np.flatnonzero(free)

    def unpack(x: np.ndarray) -> np.ndarray:
        p = np.zeros(len(REFINE_STEP))
        p[idx] = x
        return p

    score = np.inf
    for step_scale in (1.0, 0.35):
        base = camera
        simplex = np.vstack([np.zeros(len(idx)), np.diag(REFINE_STEP[idx] * step_scale)])
        res = minimize(
            lambda x, base=base: scorer.score(camera_from_local(base, unpack(x), aspect)),
            np.zeros(len(idx)),
            method="Nelder-Mead",
            options={"maxiter": iters, "xatol": 1e-5, "fatol": 1e-5, "initial_simplex": simplex},
        )
        camera, score = camera_from_local(base, unpack(res.x), aspect), float(res.fun)
    return camera, score


def main(args: Args) -> None:
    t0 = time.time()
    episodes = load_episodes(args.dataset_root)
    background, support = modal_background(episodes, "overhead")
    table_grey = table_prediction(background, support)
    chosen = select_static_frames(episodes, args.frames, seed=args.seed)
    frames = []
    for e, f in chosen:
        frames.append((episodes[e].state[f], real_dark_mask(episodes[e].image("overhead", f), table_grey)))
    print(f"{len(frames)} frames prepared in {time.time() - t0:.0f}s", flush=True)
    grasps = json.loads(args.grasps.read_text()) if args.grasps is not None else None

    if args.init is not None:
        camera = PinholeCamera.load(args.init)
    else:
        # ---- global search, quarter resolution, no lens distortion ------------------------------------
        coarse = SilhouetteScorer(frames[: args.global_frames], scale=0.25)
        if grasps is not None:
            coarse.use_grasps(grasps, args.grasp_weight)
        bounds = [(-0.9, 0.05), (-0.4, 0.4), (0.4, 1.5), (0.0, 0.7), (-0.3, 0.3), (-0.25, 0.25)]
        fy_bounds = (300.0, 900.0) if args.fy is None else (args.fy, args.fy)
        aspect_bounds = (0.7, 1.05) if args.aspect is None else (args.aspect, args.aspect)

        def with_intrinsics(p: np.ndarray) -> np.ndarray:
            # differential_evolution cannot take a zero-width bound, so pinned intrinsics are appended here
            extra = [fy_bounds[0] if args.fy is not None else p[6]]
            extra.append(aspect_bounds[0] if args.aspect is not None else p[6 + (args.fy is None)])
            return np.concatenate([p[:6], extra])

        bounds += [b for b, pinned in ((fy_bounds, args.fy), (aspect_bounds, args.aspect)) if pinned is None]
        result = differential_evolution(
            lambda p: coarse.score(camera_from_lookat(with_intrinsics(p))),
            bounds,
            seed=args.seed,
            popsize=14,
            maxiter=60,
            tol=1e-4,
            polish=False,
            init="sobol",
        )
        camera = camera_from_lookat(with_intrinsics(result.x))
        print(f"global search: score {result.fun:.2f}px after {result.nfev} evaluations", flush=True)

    free = np.ones(len(REFINE_STEP), dtype=bool)
    free[6] = args.fy is None
    free[7] = args.aspect is None
    free[10:] = args.distortion
    free[11] &= args.k2
    if args.fy is not None:
        camera.fx, camera.fy = camera.fx * args.fy / camera.fy, args.fy
    if args.aspect is not None:
        camera.fx = args.aspect * camera.fy
    score = np.inf
    for scale in args.scales:
        scorer = SilhouetteScorer(frames, scale=scale)
        if grasps is not None:
            scorer.use_grasps(grasps, args.grasp_weight)
        camera, score = refine(camera, scorer, free, args.aspect, args.iters)
        if grasps is not None:
            error = scorer.grasp_error_mm(camera)
            print(
                f"  grasp consistency: mean {error.mean():+.1f} mm, rms {np.sqrt((error**2).mean()):.1f} mm over {len(error)} events",
                flush=True,
            )
        print(
            f"refine @ {scale}: score {score:.3f}px ({time.time() - t0:.0f}s)  fx {camera.fx:.1f} fy {camera.fy:.1f} "
            f"c ({camera.cx:.1f}, {camera.cy:.1f}) k1 {camera.k1:+.3f} k2 {camera.k2:+.3f} pos {np.round(camera.pos, 4)}",
            flush=True,
        )

    camera.notes = {
        "method": "arm-silhouette chamfer fit (see calibration/overhead_camera.py)",
        "frames": len(frames),
        "score_px": score,
        "aspect_prior": args.aspect,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    camera.save(args.out)
    print(f"saved {args.out}")

    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        scorer = SilhouetteScorer(frames, scale=1.0)
        kernel = np.ones((3, 3), np.uint8)
        for i, (e, f) in enumerate(chosen[:12]):
            robot, distal = scorer.render_classes(camera, i)
            image = episodes[e].image("overhead", f)
            image[cv2.morphologyEx(robot.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0] = (0, 255, 0)
            image[cv2.morphologyEx(distal.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0] = (0, 0, 255)
            cv2.imwrite(str(args.debug_dir / f"overhead_fit_{i:02d}_{episodes[e].name}_{f:04d}.jpg"), image)


if __name__ == "__main__":
    main(tyro.cli(Args))
