"""Measure the cups, the box and where everything stood in each episode.

Each cup stands upside down on the table, so it is a truncated cone whose wide rim sits on the
table plane. Its silhouette in the calibrated overhead camera is the hull of two projected circles,
which is compared with the cup's colour mask. Size (three numbers per cup colour) is shared by every
episode; position is fitted per episode on the frame where that cup is least hidden by the arm. The
cardboard box is a cuboid fitted the same way.

A cup standing straight ahead of the arm stays hidden from the overhead camera for a whole episode.
Those are located with the wrist camera instead (once it is calibrated), and failing that from the
arm itself: every episode moves the ball from one cup to the other, so the gripper's pose when it
closed and when it opened gives both cups (``resolve_hidden_cups``).

Results go to ``results/objects.json`` (sizes; per episode, each object as the pixel it appears at
and its position on the table plane in use) and ``results/episode_starts.json`` (every episode's
starting condition, so rollouts can start from them without the recordings).

    python -m i2rt.simulation.calibration.objects
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
import tyro
from scipy.optimize import minimize

from i2rt.simulation import scene
from i2rt.simulation.calibration.common import (
    CALIB_DIR,
    DEFAULT_DATASET,
    Episode,
    load_episodes,
    robot_model,
    set_robot_state,
)
from i2rt.simulation.calibration.wrist_camera import real_masks as wrist_masks
from i2rt.simulation.cameras import PinholeCamera
from i2rt.simulation.table import TablePlane

# OpenCV HSV ranges (hue 0-180) of the three coloured objects, sampled from the footage: a strict
# "seed" range that only the object matches, and a looser "grow" range for its washed-out highlights
# (the orange cup's top disc is nearly white under the ceiling lights), kept only where it touches a seed.
HSV_RANGES = {
    "cup_orange": (((15, 150, 120), (23, 255, 255)), ((13, 45, 170), (33, 255, 255))),
    "cup_blue": (((90, 150, 90), (108, 255, 255)), ((88, 90, 70), (110, 255, 255))),
    "ball": (((4, 125, 150), (14, 255, 255)), ((3, 90, 140), (14, 255, 255))),
}
POOR_FIT = 0.2  # 1 - IoU above which a cup's overhead fit is not trusted
_PARAMS = scene.object_parameters()
BALL_CENTRE_ABOVE_TABLE = scene.ball_rest_height(_PARAMS) + _PARAMS["ball"]["radius"]
TABLE_ROI = np.array([[110, 198], [455, 198], [520, 470], [110, 470]], np.int32)  # where objects can be
_CIRCLE = np.stack(
    [np.cos(np.linspace(0, 2 * np.pi, 40, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 40, endpoint=False))],
    axis=1,
)


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    camera: Path = CALIB_DIR / "overhead_camera.json"
    wrist_camera: Path = CALIB_DIR / "wrist_camera.json"
    """Used, once it has been calibrated, for cups the overhead camera cannot see behind the arm."""
    out: Path = CALIB_DIR / "objects.json"
    debug_dir: Path | None = None


def colour_mask(image: np.ndarray, name: str) -> np.ndarray:
    """Largest blob of the object's colour on the table, as a boolean mask."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (seed_lo, seed_hi), (grow_lo, grow_hi) = HSV_RANGES[name]
    roi = np.zeros(hsv.shape[:2], np.uint8)
    cv2.fillPoly(roi, [TABLE_ROI], 255)
    seed = cv2.morphologyEx(
        cv2.inRange(hsv, np.array(seed_lo), np.array(seed_hi)) & roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed)
    if count < 2:
        return np.zeros(seed.shape, bool)
    seed = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    grow = (cv2.inRange(hsv, np.array(grow_lo), np.array(grow_hi)) & roi) > 0
    # Keep the grown pixels connected to the seed blob.
    count, labels = cv2.connectedComponents((grow | seed).astype(np.uint8))
    keep = np.unique(labels[seed])
    return np.isin(labels, keep[keep > 0])


def cup_silhouette(
    camera: PinholeCamera,
    xy: np.ndarray,
    rim: float,
    top: float,
    height: float,
    shape: tuple,
    table: TablePlane | None = None,
    parent: tuple | None = None,
) -> np.ndarray:
    """Mask of an upside-down cup (rim radius on the table, top radius at ``height``) standing at ``xy``.

    ``parent`` is the world (position, rotation) of the body the camera rides on, for the wrist camera.
    """
    table = table or TablePlane()
    normal = table.normal
    # Two unit vectors spanning the table plane.
    u = np.cross([0.0, 1.0, 0.0], normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    ring = _CIRCLE[:, :1] * u + _CIRCLE[:, 1:] * v
    base = table.point(xy)
    bottom = base + rim * ring
    upper = base + height * normal + top * ring
    projected = camera.project(np.vstack([bottom, upper]), *(parent or ()))
    mask = np.zeros(shape, np.uint8)
    if projected[:, 2].min() <= 0.01:  # partly behind the camera: no sensible outline
        return mask > 0
    pixels = np.clip(projected[:, :2], -4000, 4000)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pixels * 8).astype(np.int32)), 1, shift=3)
    return mask > 0


def overlap_loss(predicted: np.ndarray, observed: np.ndarray) -> float:
    union = (predicted | observed).sum()
    return 1.0 - (predicted & observed).sum() / max(union, 1)


def locate_cup(
    camera: PinholeCamera,
    mask: np.ndarray,
    dims: np.ndarray,
    table: TablePlane | None = None,
    parent: tuple | None = None,
) -> tuple:
    """Best (x, y) of a cup with the given dimensions for one mask. Returns (xy, loss)."""
    table = table or TablePlane()
    ys, xs = np.nonzero(mask)
    foot = np.array([[xs.mean(), np.percentile(ys, 80)]])  # low in the blob is near the rim on the table
    origin, direction = camera.pixel_ray(foot, *(parent or ()))
    start = table.intersect(origin, direction)[0, :2]
    result = minimize(
        lambda xy: overlap_loss(cup_silhouette(camera, xy, *dims, mask.shape, table, parent), mask),
        start,
        method="Nelder-Mead",
        options={"initial_simplex": start + np.array([[0, 0], [0.01, 0], [0, 0.01]]), "xatol": 2e-4, "fatol": 1e-4},
    )
    return result.x, float(result.fun)


def locate_cup_from_wrist(
    episode: Episode,
    name: str,
    dims: np.ndarray,
    table: TablePlane,
    wrist: PinholeCamera,
    count: int = 6,
    step: int = 4,
) -> tuple | None:
    """Locate a cup with the wrist camera: best (frame, xy, loss) over the frames that show most of it.

    For cups the overhead camera never sees clearly -- standing straight ahead of the arm, they hide
    behind it for a whole episode -- while the wrist camera looks right at them.
    """
    model = robot_model()
    data = mujoco.MjData(model)
    body = model.body(wrist.parent_body).id
    views = []
    for index in range(0, len(episode), step):
        mask = wrist_masks(episode.image("right_wrist", index))[name]
        # A blob cut by the image border is a cup only partly in view: its outline says little.
        if mask.sum() > 1500 and not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
            views.append((index, mask))
    views.sort(key=lambda view: -int(view[1].sum()))
    best = None
    for index, mask in views[:count]:
        set_robot_state(model, data, episode.state[index])
        parent = (data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy())
        xy, loss = locate_cup(wrist, mask, dims, table, parent)
        if best is None or loss < best[2]:
            best = (index, xy, loss)
    return best


def resolve_hidden_cups(episodes: list[Episode], result: dict, camera: PinholeCamera, table: TablePlane) -> None:
    """Place cups neither camera saw clearly from where the gripper picked up and set down the ball.

    Every episode moves the ball from one cup to the other, so the ball -- and the cup under it -- is
    in the gripper's grasp when the fingers close and again when they open. Where in the grasp is
    learned from the episodes whose cups *were* seen: the mean position of the ball in the gripper's
    frame at those moments.
    """
    model = robot_model()
    data = mujoco.MjData(model)
    gripper = model.body("gripper").id
    ball_lift = BALL_CENTRE_ABOVE_TABLE
    events: dict[str, dict] = {}
    for episode in episodes:
        closed = np.flatnonzero(episode.action[:, 6] < 0.75)
        if len(closed) < 5:
            continue
        events[episode.name] = {}
        for kind, frame in (("grasp", int(closed[0]) + 6), ("release", int(closed[-1]) - 2)):
            set_robot_state(model, data, episode.state[frame])
            events[episode.name][kind] = (data.xpos[gripper].copy(), data.xmat[gripper].reshape(3, 3).copy())

    def trusted(entry: dict | None) -> bool:
        return entry is not None and entry["loss"] <= POOR_FIT

    in_hand = []
    for name, moments in events.items():
        for position, rotation in moments.values():
            for cup in ("cup_orange", "cup_blue"):
                entry = result["episodes"][name][cup]
                if trusted(entry):
                    ball = table.point(np.array(entry["xy"]), ball_lift)
                    offset = rotation.T @ (ball - position)
                    if (
                        abs(offset[0]) < 0.05 and abs(offset[1]) < 0.05 and 0.06 < offset[2] < 0.2
                    ):  # this cup is the one in the grasp
                        in_hand.append(offset)
    if not in_hand:
        return
    mean_in_hand = np.mean(in_hand, axis=0)
    result["ball_in_gripper_at_grasp"] = [float(v) for v in mean_in_hand]
    for name, moments in events.items():
        entries = result["episodes"][name]
        spots = {}
        for kind, (position, rotation) in moments.items():
            ball = position + rotation @ mean_in_hand
            spots[kind] = (ball - table.lift_of(ball) * table.normal)[:2]  # straight down onto the table
        hidden = [cup for cup in ("cup_orange", "cup_blue") if not trusted(entries[cup])]
        if not hidden:
            continue
        other = {"cup_orange": "cup_blue", "cup_blue": "cup_orange"}
        if len(hidden) == 1:
            seen = np.array(entries[other[hidden[0]]]["xy"])
            # The seen cup is at one end of the ball's trip; the hidden one is at the other. The ball
            # does not shift in the grasp on the way, so where it sat in the gripper over the seen
            # cup -- this episode's own value, not the dataset mean -- says where the hidden cup is.
            at_grasp = np.linalg.norm(seen - spots["grasp"]) < np.linalg.norm(seen - spots["release"])
            near, far = ("grasp", "release") if at_grasp else ("release", "grasp")
            position, rotation = moments[near]
            in_hand_here = rotation.T @ (table.point(seen, ball_lift) - position)
            if np.linalg.norm(in_hand_here - mean_in_hand) < 0.04:
                position, rotation = moments[far]
                ball = position + rotation @ in_hand_here
                spots[far] = (ball - table.lift_of(ball) * table.normal)[:2]
            assignment = {hidden[0]: spots[far]}
        else:
            # Neither seen well: keep whichever pairing agrees better with the rough estimates.
            rough = {cup: np.array(entries[cup]["xy"]) if entries[cup] else None for cup in hidden}

            def cost(first: str, second: str, rough: dict = rough, spots: dict = spots) -> float:
                return sum(
                    np.linalg.norm(rough[c] - spots[k]) if rough[c] is not None else 0.1
                    for c, k in ((first, "grasp"), (second, "release"))
                )

            first, second = (
                ("cup_orange", "cup_blue")
                if cost("cup_orange", "cup_blue") <= cost("cup_blue", "cup_orange")
                else ("cup_blue", "cup_orange")
            )
            assignment = {first: spots["grasp"], second: spots["release"]}
        for cup, xy in assignment.items():
            base_px = camera.project(table.point(xy)[None])[0, :2]
            entries[cup] = {
                "xy": [float(xy[0]), float(xy[1])],
                "base_px": [float(base_px[0]), float(base_px[1])],
                "frame": None,
                "loss": None,
                "seen_by": "grasp",
            }


# ---- the cardboard box ---------------------------------------------------------------------------------

BOX_HSV = ((6, 55, 110), (18, 175, 255))  # cardboard, lit top and shaded sides
BOX_ROI = (400, 200, 565, 425)  # image region (x0, y0, x1, y1) the box is ever seen in
_BOX_CORNERS = np.array([[sx, sy, sz] for sz in (0.0, 1.0) for sx in (-0.5, 0.5) for sy in (-0.5, 0.5)])


def box_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(BOX_HSV[0]), np.array(BOX_HSV[1]))
    x0, y0, x1, y1 = BOX_ROI
    roi = np.zeros(mask.shape, np.uint8)
    roi[y0:y1, x0:x1] = 255
    mask = cv2.morphologyEx(mask & roi, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return np.zeros(mask.shape, bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def box_silhouette(
    camera: PinholeCamera, pose: np.ndarray, size: np.ndarray, shape: tuple, table: TablePlane
) -> np.ndarray:
    """Mask of a cuboid of ``size`` (width, depth, height) standing on the table at ``pose`` (x, y, yaw)."""
    c, s_ = np.cos(pose[2]), np.sin(pose[2])
    local = _BOX_CORNERS * size
    flat = np.stack([c * local[:, 0] - s_ * local[:, 1], s_ * local[:, 0] + c * local[:, 1]], axis=1)
    normal = table.normal
    u = np.cross([0.0, 1.0, 0.0], normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    corners = table.point(pose[:2]) + flat[:, :1] * u + flat[:, 1:] * v + local[:, 2:] * normal
    pixels = camera.project(corners)[:, :2]
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pixels * 8).astype(np.int32)), 1, shift=3)
    return mask > 0


def locate_box(
    camera: PinholeCamera, mask: np.ndarray, size: np.ndarray, table: TablePlane
) -> tuple[np.ndarray, float]:
    """Best (x, y, yaw) of a box of the given size for one mask, trying several headings."""
    ys, xs = np.nonzero(mask)
    start_xy = camera.pixel_to_table(np.array([[xs.mean(), np.percentile(ys, 70)]]), table)[0, :2]
    best = (None, np.inf)
    for yaw in (0.0, 0.4, 0.8, -0.4):  # a box looks the same every quarter turn
        start = np.array([start_xy[0], start_xy[1], yaw])
        result = minimize(
            lambda pose: overlap_loss(box_silhouette(camera, pose, size, mask.shape, table), mask),
            start,
            method="Nelder-Mead",
            options={
                "initial_simplex": start + np.array([[0, 0, 0], [0.015, 0, 0], [0, 0.015, 0], [0, 0, 0.15]]),
                "xatol": 3e-4,
                "fatol": 1e-4,
            },
        )
        if result.fun < best[1]:
            best = (result.x, float(result.fun))
    return best


def episode_median(episode: Episode, step: int = 6) -> np.ndarray:
    """Per-pixel median over an episode's overhead frames: the scene with the moving arm averaged away."""
    return np.median(np.stack([episode.image("overhead", i) for i in range(0, len(episode), step)]), axis=0).astype(
        np.uint8
    )


def clearest_views(episode: Episode, name: str, count: int = 6, step: int = 5) -> list[tuple[int, np.ndarray]]:
    """The frames of an episode where the object's colour blob is largest (least hidden by the arm)."""
    views = [(index, colour_mask(episode.image("overhead", index), name)) for index in range(0, len(episode), step)]
    views.sort(key=lambda view: -int(view[1].sum()))
    return views[:count]


def locate_cup_in_episode(
    camera: PinholeCamera, views: list[tuple[int, np.ndarray]], dims: np.ndarray, table: TablePlane
) -> tuple | None:
    """Best (frame, xy, loss) over an episode's clearest views of a cup.

    The largest blob is not always the cup alone (the ball is nearly the orange cup's colour, a hand
    can cover half of it), so each candidate view is fitted and the one the cup model explains best wins.
    """
    best = None
    for frame, mask in views:
        if mask.sum() < 150:
            continue
        xy, loss = locate_cup(camera, mask, dims, table)
        if best is None or loss < best[2]:
            best = (frame, xy, loss)
    return best


def main(args: Args) -> None:
    camera = PinholeCamera.load(args.camera)
    wrist = PinholeCamera.load(args.wrist_camera) if args.wrist_camera.is_file() else None
    table = TablePlane.load()
    episodes = load_episodes(args.dataset_root)
    result: dict = {"dataset": args.dataset_root.resolve().name, "cups": {}, "episodes": {}}
    views: dict[str, list] = {}
    for name in ("cup_orange", "cup_blue"):
        views[name] = [clearest_views(ep, name) for ep in episodes]
        # Fit the size on the views where the cup is whole: the biggest blobs.
        areas = np.array([v[0][1].sum() for v in views[name]])
        masks = [views[name][i][0][1] for i in np.argsort(-areas)[:14]]

        def total(dims: np.ndarray, masks: list = masks) -> float:
            if dims.min() <= 0.005:
                return 2.0
            return float(np.mean([locate_cup(camera, m, dims, table)[1] for m in masks]))

        fit = minimize(
            total,
            np.array([0.036, 0.025, 0.09]),
            method="Nelder-Mead",
            options={"xatol": 2e-4, "fatol": 2e-4, "maxiter": 120},
        )
        dims = fit.x
        result["cups"][name] = {
            "rim_radius": float(dims[0]),
            "top_radius": float(dims[1]),
            "height": float(dims[2]),
            "fit_loss": float(fit.fun),
        }
        print(
            f"{name}: rim radius {dims[0] * 1e3:.1f} mm, top radius {dims[1] * 1e3:.1f} mm, height {dims[2] * 1e3:.1f} mm (1-IoU {fit.fun:.3f})",
            flush=True,
        )

    for e, episode in enumerate(episodes):
        entry: dict = {}
        for name in ("cup_orange", "cup_blue"):
            cup = result["cups"][name]
            dims = np.array([cup["rim_radius"], cup["top_radius"], cup["height"]])
            found = locate_cup_in_episode(camera, views[name][e], dims, table)
            if found is None:
                found = (0, np.zeros(2), 1.0)  # never seen from above: leave it to the wrist camera
            frame, xy, loss = found
            source = "overhead"
            if loss > POOR_FIT and wrist is not None:
                again = locate_cup_from_wrist(episode, name, dims, table, wrist)
                if again is not None and again[2] < loss:
                    (frame, xy, loss), source = again, "right_wrist"
            # ``base_px`` -- where the centre of the cup's rim appears in the overhead image -- is the
            # measurement itself; ``xy`` is that pixel on the table plane in use when this was run.
            base_px = camera.project(table.point(xy)[None])[0, :2]
            entry[name] = {
                "xy": [float(xy[0]), float(xy[1])],
                "base_px": [float(base_px[0]), float(base_px[1])],
                "frame": frame,
                "loss": loss,
                "seen_by": source,
            }
            if loss > 0.9:
                entry[name] = None  # found by neither camera
        result["episodes"][episode.name] = entry
    # The two cups are the same product: the simulation gives both the mean of the two fits.
    result["cup"] = {
        key: float(np.mean([result["cups"][name][key] for name in ("cup_orange", "cup_blue")]))
        for key in ("rim_radius", "top_radius", "height")
    }
    resolve_hidden_cups(episodes, result, camera, table)

    # The cardboard box: one size for all episodes, a pose per episode (it was moved a few times).
    medians = [episode_median(ep) for ep in episodes]
    masks = [box_mask(m) for m in medians]
    reference = [m for m in masks[:: max(len(masks) // 6, 1)] if m.sum() > 500]

    def box_total(size: np.ndarray) -> float:
        if size.min() < 0.03:
            return 2.0
        return float(np.mean([locate_box(camera, m, size, table)[1] for m in reference]))

    fit = minimize(
        box_total,
        np.array([0.14, 0.14, 0.11]),
        method="Nelder-Mead",
        options={"xatol": 5e-4, "fatol": 3e-4, "maxiter": 80},
    )
    box_size = fit.x
    result["box"] = {"size": [float(v) for v in box_size], "fit_loss": float(fit.fun)}
    print(
        f"box: {box_size[0] * 1e3:.0f} x {box_size[1] * 1e3:.0f} x {box_size[2] * 1e3:.0f} mm (1-IoU {fit.fun:.3f})",
        flush=True,
    )
    for episode, mask in zip(episodes, masks, strict=True):
        if mask.sum() < 500:
            result["episodes"][episode.name]["box"] = None
            continue
        pose, loss = locate_box(camera, mask, box_size, table)
        base_px = camera.project(table.point(pose[:2])[None])[0, :2]
        result["episodes"][episode.name]["box"] = {
            "xy": [float(pose[0]), float(pose[1])],
            "yaw": float(pose[2]),
            "base_px": [float(base_px[0]), float(base_px[1])],
            "loss": loss,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    found = [v[n] for v in result["episodes"].values() for n in ("cup_orange", "cup_blue") if v[n]]
    losses = [c["loss"] for c in found if c["loss"] is not None]
    by = {source: sum(c["seen_by"] == source for c in found) for source in ("overhead", "right_wrist", "grasp")}
    print(
        f"located {len(found)} cups in {len(result['episodes'])} episodes: {by['overhead']} from the overhead camera, {by['right_wrist']} from the wrist camera, "
        f"{by['grasp']} from where the gripper grasped; median 1-IoU {np.median(losses):.3f}"
    )

    from i2rt.simulation import episodes as episode_data  # after objects.json exists: it reads it

    print(
        f"stored the starting conditions of {episode_data.save_starts(args.dataset_root)} episodes in {episode_data.STARTS_FILE.name}"
    )

    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        tiles = []
        for episode in episodes[::5]:
            image = episode.image("overhead", 0)
            for name, colour in (("cup_orange", (0, 0, 255)), ("cup_blue", (0, 255, 0))):
                entry = result["episodes"][episode.name][name]
                if entry is None:
                    continue
                cup = result["cups"][name]
                sil = cup_silhouette(
                    camera,
                    np.array(entry["xy"]),
                    cup["rim_radius"],
                    cup["top_radius"],
                    cup["height"],
                    image.shape[:2],
                    table,
                )
                contours, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(image, contours, -1, colour, 1)
            tiles.append(image[180:470, 130:470])
        cv2.imwrite(str(args.debug_dir / "cups.jpg"), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])]))


if __name__ == "__main__":
    main(tyro.cli(Args))
