"""Fit the wrist camera (pose on the gripper + intrinsics + lens distortion) to the recorded footage.

The wrist camera rides on the gripper, so what it sees is fixed by the arm's joint angles (logged
with every frame) and by where things are on the table (known from the calibrated overhead camera).
For a candidate camera we render what it should see and compare silhouettes with the frame:

* the two **fingertips** -- always in view, a few centimetres from the lens, sliding as the gripper
  opens and closes. They pin the camera's pose on the gripper.
* the **two cups**, seen from many distances and angles as the arm moves, and the **ball** standing
  on a cup right under the open gripper. They pin the focal length and tie the camera to the world.

The cups and the ball stand on the table, so the fit is also sensitive to where the table is in the
arm's frame; ``--fit-plane`` lets it estimate that plane along with the camera (see ``table.py``).

Recorded wrist frames were rotated 180 degrees before saving (the camera hangs upside down), so the
model fitted here is of the upright image: its principal point is the real one mirrored through the
image centre.

    python -m i2rt.simulation.calibration.wrist_camera --aspect 0.75 --no-fit-plane --plane 0.03 0 0
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
from scipy.spatial.transform import Rotation

from i2rt.simulation import scene
from i2rt.simulation.calibration.common import (
    CALIB_DIR,
    DEFAULT_DATASET,
    Episode,
    load_episodes,
    robot_model,
    set_robot_state,
)
from i2rt.simulation.cameras import PinholeCamera
from i2rt.simulation.table import TablePlane

# HSV ranges (OpenCV, hue 0-180) of the coloured objects as the wrist camera renders them. Its white
# balance differs from the overhead camera's: the "orange" cup is yellow here.
HSV = {
    # Two ranges for the yellow cup: its saturated side, and its bright, washed-out upturned base.
    # Both stay clear of the cardboard box (hue 21, saturation 90, darker).
    "cup_orange": (((23, 150, 80), (37, 255, 255)), ((27, 95, 200), (37, 255, 255))),
    # Saturated enough to exclude the lighter blue conduits and 3D-printed part in the background.
    "cup_blue": (((92, 190, 60), (108, 255, 255)),),
    "ball": (((8, 110, 140), (21, 255, 255)),),
}
CAP_PX = 60.0
_OBJECTS = scene.object_parameters()
BALL_CENTRE_ABOVE_TABLE = scene.ball_rest_height(_OBJECTS) + _OBJECTS["ball"]["radius"]
GRIPPER_CLOSING = 0.75
CANVAS = 1.35
PARENT = "gripper"


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    frames: int = 60
    global_frames: int = 20
    init: Path | None = None
    """Start the refinement from this camera instead of running the global search."""
    aspect: float | None = None
    """Pin fx/fy (0.8333 for a 16:10 sensor squashed to 4:3, 0.75 for 16:9)."""
    fit_plane: bool = True
    """Also fit the table plane (height and slopes in the robot frame)."""
    plane: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Table plane z = a + b*x + c*y to start from, or to hold when ``fit_plane`` is off."""
    scales: tuple[float, ...] = (0.5, 1.0)
    iters: int = 900
    seed: int = 0
    out: Path = CALIB_DIR / "wrist_camera.json"
    debug_dir: Path | None = None


def real_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = {}
    for name, ranges in HSV.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # Each object is one blob: keep the largest, which drops stray look-alike pixels elsewhere.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        keep = np.zeros(mask.shape, bool)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[largest, cv2.CC_STAT_AREA] > 500:
                keep = labels == largest
        masks[name] = keep
    # Fingertips: black, in the lower part of the frame, and reaching the frame's bottom edge
    # (which rules out the chairs and the dark floor that fill the top when the arm looks ahead).
    dark = (hsv[..., 2] < 45).astype(np.uint8)
    dark[:250] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    tips = np.zeros(dark.shape, bool)
    for i in range(1, count):
        touches_bottom = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= dark.shape[0] - 2
        if touches_bottom and stats[i, cv2.CC_STAT_AREA] > 300:
            tips |= labels == i
    masks["tips"] = tips
    # Cardboard (and, usefully, skin): low-saturation orange, unlike anything else on the table.
    masks["cardboard"] = (
        cv2.morphologyEx(
            cv2.inRange(hsv, np.array((15, 55, 70)), np.array((25, 150, 215))),
            cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8),
        )
        > 0
    )
    return masks


def camera_from_params(p: np.ndarray) -> PinholeCamera:
    """Camera on the gripper from: position (3), tilt, yaw, roll [rad], fy, fx/fy, cx, cy, k1.

    Angles are in the gripper mount frame (+Z forward along the fingers, +Y down, +X right): tilt
    pitches the optical axis from forward toward the fingers, yaw swings it sideways, roll spins the
    image about the optical axis.
    """
    cv_to_mount = Rotation.from_euler("y", p[4]) * Rotation.from_euler("x", -p[3]) * Rotation.from_euler("z", p[5])
    mj_to_mount = cv_to_mount.as_matrix() @ np.diag([1.0, -1.0, -1.0])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mj_to_mount.flatten())
    return PinholeCamera(
        "right_wrist",
        [float(v) for v in p[0:3]],
        [float(v) for v in quat],
        fx=float(p[6] * p[7]),
        fy=float(p[6]),
        cx=float(p[8]),
        cy=float(p[9]),
        k1=float(p[10]),
        parent_body=PARENT,
    )


class WristScorer:
    """Renders fingertips and cups through a candidate wrist camera and scores them against the frames."""

    LABELS = ("tips", "cup_orange", "cup_blue", "ball")

    def __init__(self, samples: list[dict], scale: float) -> None:
        self.scale = scale
        self.w, self.h = round(640 * scale), round(480 * scale)
        self.render_w, self.render_h = round(self.w * CANVAS), round(self.h * CANVAS)
        # A camera of its own to move around: the scene may already carry an earlier wrist calibration.
        placeholder = camera_from_params(np.array([0, -0.07, 0.07, 0.4, 0, 0, 400, 0.8, 319.5, 239.5, 0.0]))
        placeholder.name = "wrist_calibration"
        spec = scene.build_spec(scene.SceneOptions(backdrop=False))
        scene._add_camera(spec.body(PARENT), placeholder)
        self.model = spec.compile()
        self.cam_id = self.model.camera("wrist_calibration").id
        self.renderer = mujoco.Renderer(self.model, self.render_h, self.render_w)
        self.renderer.enable_segmentation_rendering()
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[scene.GROUP_VISUAL] = 1
        label_of_geom = np.zeros(self.model.ngeom + 1, np.uint8)
        for g in range(self.model.ngeom):
            body = self.model.body(self.model.geom_bodyid[g]).name
            if body in ("tip_left", "tip_right"):
                label_of_geom[g] = 1
            elif body == "cup_orange":
                label_of_geom[g] = 2
            elif body == "cup_blue":
                label_of_geom[g] = 3
            elif body == "ball":
                label_of_geom[g] = 4
            else:
                label_of_geom[g] = 5  # anything else that hides them (arm, table)
        self.label_of_geom = label_of_geom
        self.cap = CAP_PX * scale
        self.datas, self.real, self.real_dt = [], [], []
        self.samples = samples
        self.plane = TablePlane()
        for sample in samples:
            data = mujoco.MjData(self.model)
            set_scene_state(self.model, data, sample, self.plane)
            self.datas.append(data)
            small = {
                k: cv2.resize(v.astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST) > 0
                for k, v in sample["masks"].items()
            }
            self.real.append(small)
            # For "rendered pixel should land on the real object", anything that can hide the object
            # in the real frame counts as a hit too: a finger hides anything, the ball hides a cup.
            hides = {"tips": small["tips"], "ball": small["ball"] | small["tips"]}
            self.real_dt.append(
                {k: self._dt(hides.get(k, small[k] | small["ball"] | small["tips"])) for k in self.LABELS}
            )
        self._maps_key, self._maps = None, None

    def _dt(self, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.full(mask.shape, self.cap, np.float32)
        return np.minimum(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3), self.cap)

    def render_labels(self, camera: PinholeCamera, index: int) -> np.ndarray:
        camera.apply_to_model(self.model, self.cam_id, canvas_scale=CANVAS)
        data = self.datas[index]
        mujoco.mj_camlight(self.model, data)  # the renderer reads the camera pose from data
        self.renderer.update_scene(data, camera=self.cam_id, scene_option=self.option)
        seg = self.renderer.render()
        geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], self.model.ngeom)
        labels = self.label_of_geom[np.clip(geom, 0, self.model.ngeom)]
        key = (camera.fx, camera.fy, camera.cx, camera.cy, camera.k1, camera.k2)
        if key != self._maps_key:
            self._maps = camera.distortion_maps(self.render_w, self.render_h, CANVAS, out_scale=self.scale)
            self._maps_key = key
        return cv2.remap(labels, self._maps[0], self._maps[1], cv2.INTER_NEAREST, borderValue=0)

    def set_plane(self, plane: TablePlane) -> None:
        """Move the cups (and a ball standing on one) onto the table plane."""
        if plane == self.plane:
            return
        self.plane = plane
        for sample, data in zip(self.samples, self.datas, strict=True):
            set_scene_state(self.model, data, sample, plane)

    def score(self, camera: PinholeCamera, plane: TablePlane | None = None) -> float:
        if plane is not None:
            self.set_plane(plane)
        total, weight = 0.0, 0.0
        for i in range(len(self.datas)):
            labels = self.render_labels(camera, i)
            for value, name in enumerate(self.LABELS, start=1):
                sim, real = labels == value, self.real[i][name]
                if name == "ball" and self.samples[i]["ball_on"] is None:
                    continue  # in the gripper: where exactly is not known, so it is not scored
                if not sim.any() and not real.any():
                    continue
                forward = self.real_dt[i][name][sim].mean() if sim.any() else self.cap
                backward = (
                    self._dt(sim)[real].mean()
                    if real.any()
                    else 0.0
                    if name in ("cup_orange", "cup_blue")
                    else self.cap
                )
                w = 1.5 if name == "tips" else 1.0
                total += w * (forward + backward) / (2.0 * self.scale)
                weight += w
        return total / max(weight, 1e-9)


def set_scene_state(model: mujoco.MjModel, data: mujoco.MjData, sample: dict, plane: TablePlane) -> None:
    """Arm at the recorded state; cups where the overhead camera saw them on the table plane; the ball
    on the cup it was standing on at that moment, or out of sight while the gripper held it."""
    set_robot_state(model, data, sample["state"])
    where = {}
    for name in ("cup_orange", "cup_blue"):
        adr = model.joint(f"{name}_free").qposadr[0]
        ray = sample["cup_rays"].get(name)
        where[name] = (
            plane.intersect(np.asarray(ray[0]), np.asarray(ray[1]))[0]
            if ray is not None
            else np.array([3.0, 3.0, 0.0])
        )
        data.qpos[adr : adr + 3] = where[name]
        data.qpos[adr + 3 : adr + 7] = plane.quat
    adr = model.joint("ball_free").qposadr[0]
    if sample["ball_on"] is not None:
        data.qpos[adr : adr + 3] = where[sample["ball_on"]] + BALL_CENTRE_ABOVE_TABLE * plane.normal
    else:
        data.qpos[adr : adr + 3] = [3.0, -3.0, 0.5]
    mujoco.mj_forward(model, data)


def select_samples(episodes: list[Episode], objects: dict, count: int, seed: int) -> list[dict]:
    """Still frames spread over gripper poses and openings.

    Half are close views of the ball standing on a cup (the gripper hovering over it before a grasp
    or after a release): they show the table's height under the arm at close range. The rest have a
    cup in view from farther away.
    """
    model = robot_model()
    data = mujoco.MjData(model)
    site = model.site("grasp_site").id
    overhead = PinholeCamera.load(CALIB_DIR / "overhead_camera.json")
    located_on = TablePlane.load()  # the plane ``objects.json`` was produced with
    candidates = []
    for episode in episodes:
        cups, cup_xy = {}, {}
        for k, v in objects["episodes"][episode.name].items():
            if v:  # the overhead camera's ray to the cup's base, whatever the table height turns out to be
                pixel = overhead.project(located_on.point(np.array(v["xy"]))[None])[:, :2]
                origin, direction = overhead.pixel_ray(pixel)
                cups[k] = (origin.tolist(), direction[0].tolist())
                cup_xy[k] = np.array(v["xy"])
        s = episode.state
        closed = np.flatnonzero(episode.action[:, 6] < GRIPPER_CLOSING)
        grasp, release = (int(closed[0]), int(closed[-1])) if len(closed) >= 5 else (None, None)
        source = target = None
        if grasp is not None and cup_xy:
            for label, frame in (("source", grasp + 6), ("target", release)):
                set_robot_state(model, data, s[min(frame, len(s) - 1)])
                nearest = min(cup_xy, key=lambda name: np.linalg.norm(cup_xy[name] - data.site_xpos[site][:2]))
                source, target = (nearest, target) if label == "source" else (source, nearest)
        speed = np.abs(s[6:, :] - s[:-6, :]).max(axis=1)
        for t in range(3, len(s) - 3, 4):
            if speed[t - 3] > 0.02:
                continue
            if grasp is None:
                ball_on = None
            elif t < grasp - 3:
                ball_on = source
            elif t > release + 8:
                ball_on = target
            else:
                ball_on = None
            masks = real_masks(episode.image("right_wrist", t))
            cup_area = masks["cup_orange"].sum() + masks["cup_blue"].sum()
            set_robot_state(model, data, s[t])
            feature = np.concatenate(
                [data.site_xpos[site], 0.2 * data.site_xmat[site].reshape(3, 3)[:, 2], [0.3 * s[t, 6]]]
            )
            candidates.append(
                {"episode": episode.name, "frame": t, "state": s[t].copy(), "cup_rays": cups, "masks": masks, "feature": feature,
                 "cup_area": cup_area, "ball_on": ball_on, "ball_area": int(masks["ball"].sum())}
            )  # fmt: skip

    def spread(pool: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
        if len(pool) <= n:
            return pool
        features = np.array([c["feature"] for c in pool])
        chosen = [int(rng.integers(len(pool)))]
        dist = np.linalg.norm(features - features[chosen[0]], axis=1)
        while len(chosen) < n:
            nxt = int(dist.argmax())
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(features - features[nxt], axis=1))
        return [pool[i] for i in chosen]

    rng = np.random.default_rng(seed)
    close_ball = [c for c in candidates if c["ball_on"] is not None and c["ball_area"] > 6000]
    with_cup = [
        c for c in candidates if c["cup_area"] > 1200 and not (c["ball_on"] is not None and c["ball_area"] > 6000)
    ]
    near = spread(close_ball, count // 2, rng)
    far = spread(with_cup if len(with_cup) >= count - len(near) else candidates, count - len(near), rng)
    # Interleave so any leading subset (the global search uses one) holds both kinds.
    mixed = [x for pair in zip(near, far, strict=False) for x in pair] + near[len(far) :] + far[len(near) :]
    return mixed


# Global-search box: position on the gripper [m], tilt/yaw/roll [rad], fy [px], fx/fy.
GLOBAL_BOUNDS = [
    (-0.03, 0.03),
    (-0.13, -0.03),
    (-0.03, 0.12),
    (0.0, 0.9),
    (-0.15, 0.15),
    (-0.12, 0.12),
    (250.0, 750.0),
    (0.7, 0.9),
]
# Refinement parameters: the eleven camera parameters, then the table plane z = a + b*x + c*y in the
# robot frame (see table.py). The cups stand on that plane, so the wrist camera -- whose pose on the
# gripper the fingertips already fix -- measures how high the table really is under the arm.
REFINE_STEP = np.array([0.004, 0.004, 0.004, 0.02, 0.02, 0.02, 12.0, 0.02, 8.0, 8.0, 0.05, 0.008, 0.015, 0.015])
N_CAMERA = 11


def main(args: Args) -> None:
    t0 = time.time()
    episodes = load_episodes(args.dataset_root)
    objects = json.loads((CALIB_DIR / "objects.json").read_text())
    samples = select_samples(episodes, objects, args.frames, args.seed)
    print(f"{len(samples)} frames selected in {time.time() - t0:.0f}s", flush=True)

    def full(p: np.ndarray) -> np.ndarray:
        q = np.asarray(p, dtype=float).copy()
        if args.aspect is not None:
            q[7] = args.aspect
        return q

    if args.init is not None:
        start = json.loads(args.init.read_text())["notes"]
        plane0 = (list(start.get("table_plane", [])) + [0.0, 0.0, 0.0])[:3] if args.fit_plane else list(args.plane)
        params = np.concatenate([np.array(start["params"], dtype=float)[:N_CAMERA], plane0])
    else:
        coarse = WristScorer(samples[: args.global_frames], scale=0.25)
        coarse.set_plane(TablePlane(*args.plane))
        result = differential_evolution(
            lambda p: coarse.score(camera_from_params(full(np.concatenate([p, [319.5, 239.5, 0.0]])))),
            GLOBAL_BOUNDS,
            seed=args.seed,
            popsize=12,
            maxiter=70,
            tol=1e-4,
            polish=False,
            init="sobol",
        )
        params = np.concatenate([full(np.concatenate([result.x, [319.5, 239.5, 0.0]])), args.plane])
        print(
            f"global search: score {result.fun:.2f}px after {result.nfev} evaluations ({time.time() - t0:.0f}s)\n  {np.round(params, 4).tolist()}",
            flush=True,
        )

    free = np.ones(len(REFINE_STEP), bool)
    free[7] = args.aspect is None
    free[N_CAMERA:] = args.fit_plane
    idx = np.flatnonzero(free)
    score = np.inf
    for scale in args.scales:
        scorer = WristScorer(samples, scale=scale)
        for step_scale in (1.0, 0.35):
            base = params.copy()

            def unpack(x: np.ndarray, base: np.ndarray = base) -> np.ndarray:
                q = base.copy()
                q[idx] += x
                return full(q)

            res = minimize(
                lambda x, scorer=scorer, unpack=unpack: scorer.score(
                    camera_from_params(unpack(x)[:N_CAMERA]), TablePlane(*unpack(x)[N_CAMERA:])
                ),
                np.zeros(len(idx)),
                method="Nelder-Mead",
                options={
                    "maxiter": args.iters,
                    "xatol": 1e-5,
                    "fatol": 1e-5,
                    "initial_simplex": np.vstack([np.zeros(len(idx)), np.diag(REFINE_STEP[idx] * step_scale)]),
                },
            )
            params, score = unpack(res.x), float(res.fun)
        print(
            f"refine @ {scale}: score {score:.3f}px ({time.time() - t0:.0f}s)\n  {np.round(params, 4).tolist()}",
            flush=True,
        )

    camera = camera_from_params(params[:N_CAMERA])
    camera.notes = {
        "table_plane": [float(v) for v in params[N_CAMERA:]],
        "table_plane_meaning": "table surface z = a + b*x + c*y in the robot base frame, as seen by this camera",
        "method": "fingertip + cup silhouette fit (see calibration/wrist_camera.py)",
        "frames": len(samples),
        "score_px": score,
        "params": [float(v) for v in params],
        "params_meaning": "x, y, z on the gripper [m]; tilt, yaw, roll [rad]; fy [px]; fx/fy; cx, cy [px]; k1",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    camera.save(args.out)
    print(f"saved {args.out}")

    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        scorer = WristScorer(samples, scale=1.0)
        scorer.set_plane(TablePlane(*params[N_CAMERA:]))
        kernel = np.ones((3, 3), np.uint8)
        episodes_by_name = {e.name: e for e in episodes}
        for i, sample in enumerate(samples[:16]):
            labels = scorer.render_labels(camera, i)
            image = episodes_by_name[sample["episode"]].image("right_wrist", sample["frame"])
            for value, colour in ((1, (0, 255, 0)), (2, (0, 0, 255)), (3, (255, 0, 255)), (4, (255, 255, 0))):
                image[cv2.morphologyEx((labels == value).astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0] = colour
            cv2.imwrite(
                str(args.debug_dir / f"wrist_fit_{i:02d}_{sample['episode']}_{sample['frame']:04d}.jpg"), image
            )


if __name__ == "__main__":
    main(tyro.cli(Args))
