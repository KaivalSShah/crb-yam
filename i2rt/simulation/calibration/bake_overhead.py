"""Turn the overhead camera's footage into the surfaces that camera sees in simulation.

The overhead camera never moves, so everything static in its view -- table, wall, poster, carpet,
window -- is one fixed picture: the per-pixel most common colour over the whole dataset (the arm,
cups, ball and operator average away). On the table, where the arm works, that is not left to
chance: the simulation knows where the arm and the objects were in every frame, so each table pixel
is the median of the frames in which nothing simulated covered it (``unoccluded_median``). The
picture is re-projected onto two pieces of geometry:

* the **table top**, as a polygon in the table plane. It is a real surface the arm and the objects
  stand on, so it is drawn lit: simulated shadows fall on a photograph of the real table.
* a **backdrop** sheet far behind the scene that fills the rest of the camera's view.

Seen through the calibrated overhead camera the two reproduce the real background pixel for pixel.

    python -m i2rt.simulation.calibration.bake_overhead
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
from i2rt.simulation.calibration.common import (
    CALIB_DIR,
    DEFAULT_DATASET,
    SIM_ROOT,
    load_episodes,
    median_background,
    modal_background,
)
from i2rt.simulation.calibration.objects import box_mask, episode_median
from i2rt.simulation.cameras import PinholeCamera
from i2rt.simulation.env import EnvConfig, YamStationEnv
from i2rt.simulation.mesh_utils import grid_sheet, save_obj
from i2rt.simulation.table import TablePlane

TEXEL = 0.002  # metres per table texel
BACKDROP_DEPTH = 6.0  # metres from the camera; anything beyond the room works
BACKDROP_UPSCALE = 2
# Image region where the follower arm hides the table in most frames (around its base and upper arm).
ARM_GHOST_POLYGON = [[292, 262], [376, 262], [384, 479], [286, 479]]
# Image region the follower arm sweeps over while working (cups, ball and its own shadow included).
REACH_POLYGON = [[212, 204], [422, 204], [436, 479], [212, 479]]  # stops short of the cardboard box
# Image region the operator's arm, the leader arm and a mouse occupy on and off.
OPERATOR_POLYGON = [[0, 232], [150, 232], [150, 400], [132, 479], [0, 479]]
MIN_VIEWS = 16  # frames a table pixel must be seen uncovered in before its median is believed
VIEW_MARGIN = 10  # pixels left out around everything simulated: calibration residual, and the core of its shadow


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    camera: Path = CALIB_DIR / "overhead_camera.json"
    out_layout: Path = CALIB_DIR / "overhead_layout.json"
    assets: Path = SIM_ROOT / "assets"
    debug_dir: Path | None = None


def fit_line(points: np.ndarray, tol: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """Robust image line through ``points`` (N, 2): returns (point on line, unit direction)."""
    best, best_count = None, -1
    rng = np.random.default_rng(0)
    for _ in range(400):
        i, j = rng.choice(len(points), size=2, replace=False)
        d = points[j] - points[i]
        if np.linalg.norm(d) < 20:
            continue
        d = d / np.linalg.norm(d)
        dist = np.abs((points - points[i]) @ np.array([-d[1], d[0]]))
        count = int((dist < tol).sum())
        if count > best_count:
            best, best_count = dist < tol, count
    inliers = points[best]
    centre = inliers.mean(axis=0)
    direction = np.linalg.svd(inliers - centre)[2][0]
    return centre, direction


def table_edges(plate: np.ndarray, support: np.ndarray) -> tuple[tuple, tuple]:
    """The table's far edge and right edge as image lines, found as brightness steps in the plate."""
    grey = cv2.GaussianBlur(plate.mean(axis=2), (0, 0), 1.0)

    def step(profile: np.ndarray) -> tuple[float, float]:
        d = np.diff(profile)
        i = int(np.argmax(d))
        offset = 0.0
        if 0 < i < len(d) - 1:
            a, b, c = d[i - 1], d[i], d[i + 1]
            offset = 0.5 * (a - c) / (a - 2 * b + c) if abs(a - 2 * b + c) > 1e-6 else 0.0
        return i + 0.5 + offset, float(d[i])

    far = []
    for x in range(60, 440, 3):  # carpet/chair above -> brighter table below
        if support[183:212, x].min() < 0.7:
            continue
        position, strength = step(grey[183:212, x])
        if strength > 8:
            far.append((x, 183 + position))
    right = []
    for y in range(215, 472, 3):  # table -> bright partition to its right
        position, strength = step(grey[y, 440:580])
        if strength > 12:
            right.append((440 + position, y))
    return fit_line(np.array(far)), fit_line(np.array(right))


def smooth_fill(
    image: np.ndarray, weight: np.ndarray, sigmas: tuple[float, ...] = (12.0, 30.0, 70.0, 160.0)
) -> np.ndarray:
    """Smooth field through the trusted pixels of ``image`` (normalised convolution, coarse to fine).

    Each pixel takes the finest scale at which enough trusted pixels are in range, so narrow gaps keep
    local detail and wide ones still get filled.
    """
    result = np.zeros_like(image, dtype=np.float32)
    filled = np.zeros(weight.shape, bool)
    for sigma in sigmas:
        den = cv2.GaussianBlur(weight, (0, 0), sigma)
        num = cv2.GaussianBlur(image * weight[..., None], (0, 0), sigma)
        use = (den > 0.15) & ~filled
        result[use] = num[use] / den[use][:, None]
        filled |= use
    result[~filled] = (image * weight[..., None]).sum(axis=(0, 1)) / max(float(weight.sum()), 1.0)
    return cv2.GaussianBlur(result, (0, 0), 4.0)


def relax(field: np.ndarray, image: np.ndarray, known: np.ndarray, iterations: int = 300) -> np.ndarray:
    """Smoothest surface through the ``known`` pixels of ``image``, relaxed from ``field``.

    ``smooth_fill`` switches scale part-way into a wide gap, which leaves a faint step there. Diffusing
    its result while holding the known pixels fixed irons that out: across the patch the arm always
    covers, the table then ramps evenly from its bright side to its dark one.
    """
    fixed = known[..., None]
    result = np.where(fixed, image, field).astype(np.float32)
    for _ in range(iterations):
        result = np.where(fixed, image, cv2.blur(result, (9, 9)))
    return result


def unoccluded_median(dataset_root: Path, step: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median over the frames in which nothing simulated covers the pixel, and that frame count.

    The simulation knows where the arm, the cups, the ball and the box were in every recorded frame,
    so there is no need to hope they average away: each frame's copy of them is left out, a few pixels
    wider for the calibration residual and the darkest part of their shadows. What remains at a pixel
    is the table as the camera really saw it, scuff marks included -- also under the arm's reach,
    where the dataset-wide mode is a patchwork of table, arm, cups and shade.
    """
    episodes = episode_data.load(dataset_root)
    setups = episode_data.all_setups(dataset_root)
    # No photographed surfaces: this runs before they exist, and only the outlines are wanted.
    env = YamStationEnv(EnvConfig(cameras=("overhead",), scene=scene.SceneOptions(backdrop=False)))
    env.render("overhead")
    renderer = env._renderers["overhead"]
    lag = renderer.image_model.latency
    simulated = np.append(env.model.geom_bodyid != 0, False)  # the arm's and the objects' geoms; last entry: "nothing"
    kernel = np.ones((2 * VIEW_MARGIN + 1, 2 * VIEW_MARGIN + 1), np.uint8)
    frames, hidden = [], []
    for name, episode in episodes.items():
        for index in range(0, len(episode), step):
            pose_scene(env, episode, setups[name], index, lag=lag)
            covered = simulated[renderer.render_segmentation(env.data)]
            hidden.append(cv2.dilate(covered.astype(np.uint8), kernel) > 0)
            frames.append(episode.image("overhead", index))
    env.close()
    frames, hidden = np.stack(frames), np.stack(hidden)
    views = (~hidden).sum(axis=0)
    median = np.zeros(frames.shape[1:], np.float32)
    for rows in np.array_split(np.arange(frames.shape[1]), 16):  # in bands, to bound the memory
        values = np.ma.masked_array(frames[:, rows], mask=np.repeat(hidden[:, rows, :, None], 3, axis=3))
        median[rows] = np.ma.median(values, axis=0).filled(0)
    return median, views


def intersect(line_a: tuple, line_b: tuple) -> np.ndarray:
    (pa, da), (pb, db) = line_a, line_b
    t = np.linalg.solve(np.array([da, -db]).T, pb - pa)
    return pa + t[0] * da


def at_y(line: tuple, y: float) -> np.ndarray:
    p, d = line
    return p + d * (y - p[1]) / d[1]


def at_x(line: tuple, x: float) -> np.ndarray:
    p, d = line
    return p + d * (x - p[0]) / d[0]


def main(args: Args) -> None:
    camera = PinholeCamera.load(args.camera)
    episodes = load_episodes(args.dataset_root)
    plate, support = modal_background(episodes, "overhead")
    height, width = plate.shape[:2]
    textures = args.assets / "textures"
    meshes = args.assets / "meshes"
    textures.mkdir(parents=True, exist_ok=True)
    meshes.mkdir(parents=True, exist_ok=True)

    # ---- where the table is ---------------------------------------------------------------------
    far, right = table_edges(plate, support)
    corner = intersect(far, right)
    polygon_px = np.array([at_x(far, -0.5), corner, at_y(right, height - 0.5), [-0.5, height - 0.5]])
    table = TablePlane.load()
    polygon_world = camera.pixel_to_table(polygon_px, table)[:, :2]
    table_mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(table_mask, [np.round(polygon_px).astype(np.int32)], 1)

    # ---- clean plate ------------------------------------------------------------------------------
    # Off the table the scene itself changed between episodes (chairs, people), which the per-pixel
    # mode renders as speckle; the median of episode medians keeps those regions coherent.
    median = median_background(episodes, "overhead")
    plate = np.where(table_mask[..., None] > 0, plate, median)
    # On the table, wherever a pixel was seen uncovered often enough, that is the table itself. (Not at
    # the operator's end: nothing simulated stands in for the operator, so nothing is left out there.)
    operator_zone = np.zeros((height, width), np.uint8)
    cv2.fillPoly(operator_zone, [np.array(OPERATOR_POLYGON, np.int32)], 1)
    seen, views = unoccluded_median(args.dataset_root)
    reliable = (views >= MIN_VIEWS) & (table_mask > 0) & (operator_zone == 0)
    plate = np.where(reliable[..., None], seen, plate)
    # What is left are the pixels that never, or hardly ever, show table: under the arm's base, under
    # the box, and the operator's end. The "most common colour" there is whatever covers them. Paint
    # the table back in from the smooth brightness field of the pixels around (what the table has
    # there cannot be known; its soft sheen can be continued).
    grey = plate.mean(axis=2)
    saturation = cv2.cvtColor(np.clip(plate, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)[..., 1]
    # Bare table: seen in most frames, mid-grey, and colourless (which the box and the cups' traces are not).
    trusted = (((support > 0.75) | reliable) & (table_mask > 0) & (grey > 95) & (saturation < 45)).astype(np.float32)
    field = smooth_fill(plate, trusted)
    arm_zone = np.zeros((height, width), np.uint8)
    cv2.fillPoly(arm_zone, [np.array(ARM_GHOST_POLYGON, np.int32)], 1)
    off_table_colour = np.abs(plate - field).max(axis=2) > 28
    # ...except what is always there and simply is not table: the arm's base plate and the cables
    # around it. They stay in the picture (the base stands on the table; drawn as part of it, it
    # stays put however the table plane is set relative to the arm's kinematics).
    fixture = (support >= 0.85) & off_table_colour
    ghost = ((arm_zone > 0) & (table_mask > 0) & (off_table_colour | (support < 0.5)) & ~fixture).astype(np.uint8)
    # The operator and the leader arm work over the table's left end. They are not part of the
    # simulated scene, and what the mode leaves of them is a smear, so that end is painted clean too.
    ghost |= ((operator_zone > 0) & (table_mask > 0) & (off_table_colour | (support < 0.8))).astype(np.uint8)
    # Across the arm's reach the table is clear in most frames, but shaded differently from frame to
    # frame by the arm above it, and the cups stand there. Where no single appearance clearly wins,
    # the mode is a patchwork of those; the smooth field is closer to the bare table than any of them.
    reach_zone = np.zeros((height, width), np.uint8)
    cv2.fillPoly(reach_zone, [np.array(REACH_POLYGON, np.int32)], 1)
    ghost |= ((reach_zone > 0) & (table_mask > 0) & (support < 0.86)).astype(np.uint8)
    # The cardboard box is a simulated object in its own right (it gets moved): take its picture off the table.
    ghost |= cv2.dilate(box_mask(np.clip(plate, 0, 255).astype(np.uint8)).astype(np.uint8), np.ones((9, 9), np.uint8))
    ghost[reliable] = 0  # a pixel seen uncovered is the table, whatever the mode made of it
    ghost = cv2.dilate(ghost, np.ones((9, 9), np.uint8))
    feather = cv2.GaussianBlur(ghost.astype(np.float32), (0, 0), 2.5)[..., None]
    field = relax(field, plate, (trusted > 0) & (ghost == 0))
    clean = np.clip(plate * (1 - feather) + field * feather, 0, 255).astype(np.uint8)
    plate_u8 = np.clip(plate, 0, 255).astype(np.uint8)

    # ---- table texture: the plate re-projected onto z = 0 -------------------------------------------
    lo = polygon_world.min(axis=0) - 0.01
    hi = polygon_world.max(axis=0) + 0.01
    cols = int(np.ceil((hi[1] - lo[1]) / TEXEL))  # texture u runs along world -Y (left to right in the image)
    rows = int(np.ceil((hi[0] - lo[0]) / TEXEL))  # texture v runs along world -X (far to near)
    world_y = hi[1] - (np.arange(cols) + 0.5) * TEXEL
    world_x = hi[0] - (np.arange(rows) + 0.5) * TEXEL
    grid_x, grid_y = np.meshgrid(world_x, world_y, indexing="ij")
    grid = np.stack([grid_x, grid_y, table.height(grid_x, grid_y)], axis=-1)
    uv = camera.project(grid.reshape(-1, 3))[:, :2].reshape(rows, cols, 2).astype(np.float32)
    table_texture = cv2.remap(clean, uv[..., 0], uv[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(str(textures / "overhead_table.png"), table_texture)

    def table_uv(xy: np.ndarray) -> np.ndarray:
        return np.stack([(hi[1] - xy[:, 1]) / (cols * TEXEL), (hi[0] - xy[:, 0]) / (rows * TEXEL)], axis=1)

    vertices = np.hstack([polygon_world, table.height(polygon_world[:, 0], polygon_world[:, 1])[:, None]])
    order = np.arange(len(vertices))
    # Counter-clockwise seen from above, so the sheet faces up.
    signed_area = 0.5 * np.sum(
        polygon_world[:, 0] * np.roll(polygon_world[:, 1], -1) - np.roll(polygon_world[:, 0], -1) * polygon_world[:, 1]
    )
    if signed_area < 0:
        order = order[::-1]
    faces = np.array([[order[0], order[i], order[i + 1]] for i in range(1, len(order) - 1)])
    save_obj(meshes / "overhead_table.obj", vertices, faces, table_uv(polygon_world))

    # ---- backdrop: the whole plate on a sheet that fills the view -----------------------------------
    backdrop = cv2.resize(clean, None, fx=BACKDROP_UPSCALE, fy=BACKDROP_UPSCALE, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(textures / "overhead_backdrop.png"), backdrop)
    # Beyond the table the room changed between episodes (a chair was moved, people came and went).
    # One backdrop per episode -- that episode's own median frame around the common, cleaned table --
    # lets a replayed episode be compared with its own surroundings (``YamStationEnv.set_backdrop``).
    per_episode = args.assets / "backdrops" / "overhead"
    per_episode.mkdir(parents=True, exist_ok=True)
    keep_common = cv2.GaussianBlur(cv2.dilate(table_mask, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.5)[
        ..., None
    ]
    for episode in episodes:
        own = episode_median(episode).astype(np.float32)
        cv2.imwrite(
            str(per_episode / f"{episode.name}.jpg"),
            (clean * keep_common + own * (1 - keep_common)).astype(np.uint8),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

    px, py = np.meshgrid(np.linspace(-0.5, width - 0.5, 33), np.linspace(-0.5, height - 0.5, 25))
    origin, rays = camera.pixel_ray(np.stack([px.ravel(), py.ravel()], axis=1))
    rotation, _ = camera.world_to_cv()
    depth = rays @ rotation[2]  # component along the optical axis
    points = origin + rays * (BACKDROP_DEPTH / depth)[:, None]
    sheet_uv = np.stack([(px + 0.5) / width, (py + 0.5) / height], axis=-1)
    save_obj(meshes / "overhead_backdrop.obj", *grid_sheet(points.reshape(25, 33, 3), sheet_uv))

    layout = {
        "table_polygon_px": polygon_px.tolist(),
        "table_polygon_world": polygon_world.tolist(),
        "table_far_edge_x": float(polygon_world[:2, 0].mean()),
        "table_right_edge_y": float(polygon_world[1:3, 1].mean()),
        "table_texture": {
            "file": "textures/overhead_table.png",
            "x_range": [float(lo[0]), float(hi[0])],
            "y_range": [float(lo[1]), float(hi[1])],
            "texel": TEXEL,
        },
        "table_mesh": "meshes/overhead_table.obj",
        "backdrop_texture": "textures/overhead_backdrop.png",
        "backdrop_mesh": "meshes/overhead_backdrop.obj",
        "notes": "generated by calibration/bake_overhead.py from the overhead camera's modal background",
    }
    args.out_layout.write_text(json.dumps(layout, indent=2) + "\n")
    print(f"table polygon (world xy): {np.round(polygon_world, 3).tolist()}")
    print(f"table texture {cols}x{rows}px, backdrop {backdrop.shape[1]}x{backdrop.shape[0]}px -> {textures}")
    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        vis = clean.copy()
        cv2.polylines(vis, [np.round(polygon_px).astype(np.int32)], True, (0, 0, 255), 1)
        contours, _ = cv2.findContours(ghost, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(str(args.debug_dir / "overhead_plate_clean.jpg"), np.hstack([plate_u8, vis]))
        uncovered = np.where((views >= MIN_VIEWS)[..., None], seen, 0).astype(np.uint8)
        cv2.imwrite(str(args.debug_dir / "overhead_unoccluded.jpg"), uncovered)


if __name__ == "__main__":
    main(tyro.cli(Args))
