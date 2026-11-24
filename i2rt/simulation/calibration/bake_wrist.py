"""Build the room the wrist camera sees, from the wrist camera's own footage.

The wrist camera moves with the gripper, so unlike the overhead camera it needs real 3D
surroundings. The room is approximated by four flat surfaces placed from the overhead camera's
view of it -- the table top, the floor, the wall behind the table and the partition to its right --
and each is textured by the wrist camera itself: every recorded wrist frame, with its camera pose
known from the logged joint angles, is back-projected onto the surface its pixels look at, skipping
whatever the arm, the cups or the ball cover. Per texel the most common colour over all frames is
kept, so things that moved (the operator, a chair, the ball) fade out.

The result is a wrist view whose table shows its real scuff marks at close range and whose
background shows the real room in roughly the right place with roughly the right parallax.

    python -m i2rt.simulation.calibration.bake_wrist
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
from i2rt.simulation.calibration.common import CACHE_DIR, CALIB_DIR, DEFAULT_DATASET, SIM_ROOT
from i2rt.simulation.calibration.wrist_camera import real_masks
from i2rt.simulation.env import EnvConfig, YamStationEnv
from i2rt.simulation.mesh_utils import save_obj
from i2rt.simulation.table import TablePlane

TABLE_HEIGHT = 0.74  # table top above the floor [m]: a standard desk; the overhead view cannot measure it
WALL_BEYOND_FAR_EDGE = (
    1.45  # back wall behind the table's far edge [m], from the overhead camera's view of the skirting board
)
PARTITION_BEYOND_RIGHT_EDGE = 0.075  # partition wall next to the table's right edge [m], likewise
BINS = 12  # grey-level histogram bins used to find each texel's most common appearance
PER_EPISODE_SCALE = 0.6  # resolution of the per-episode wall/floor pictures relative to the shared ones


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    every: int = 6
    table_smoothing: float = 0.015
    """Gaussian smoothing of the table texture [m]."""
    detail_every: int = 2
    """Of those, use every n-th for the walls and floor (they need far fewer frames than the table)."""
    """Use every n-th frame of every episode."""
    out_layout: Path = CALIB_DIR / "wrist_layout.json"
    assets: Path = SIM_ROOT / "assets"


@dataclass
class Sheet:
    """A flat rectangle in the world: ``p0`` plus the edge vectors ``eu`` and ``ev``; texture is (rows along ev) x (cols along eu)."""

    name: str
    p0: np.ndarray
    eu: np.ndarray
    ev: np.ndarray
    texel: float
    lit: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        return int(np.ceil(np.linalg.norm(self.ev) / self.texel)), int(np.ceil(np.linalg.norm(self.eu) / self.texel))

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.eu, self.ev)
        return n / np.linalg.norm(n)

    def hit(self, origin: np.ndarray, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Distance along each ray to the sheet (inf where it misses) and the (u, v) hit coordinates in [0, 1]."""
        n = self.normal
        denom = rays @ n
        t = ((self.p0 - origin) @ n) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        rel = origin + t[:, None] * rays - self.p0
        u = rel @ self.eu / (self.eu @ self.eu)
        v = rel @ self.ev / (self.ev @ self.ev)
        ok = (t > 0.02) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        return np.where(ok, t, np.inf), u, v


def bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Colours of ``image`` at fractional pixel positions (all inside the image)."""
    x0 = np.clip(np.floor(x).astype(int), 0, image.shape[1] - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, image.shape[0] - 2)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return np.clip(top * (1 - fy) + bottom * fy, 0, 255).astype(np.uint8)


def room_sheets(table: TablePlane) -> list[Sheet]:
    """The four surfaces, laid out from the table's edges as the overhead camera located them."""
    polygon = np.array(scene.load_json("overhead_layout.json")["table_polygon_world"])
    on_table = [table.point(p) for p in polygon]  # far-left, far-right corner, near-right, near-left
    corner = on_table[1]
    along_far = (on_table[0] - corner) / np.linalg.norm(on_table[0] - corner)  # towards the table's left end
    towards_near = (on_table[2] - corner) / np.linalg.norm(on_table[2] - corner)  # towards the operator
    up = table.normal
    floor_corner = corner - TABLE_HEIGHT * up
    return [
        Sheet("wrist_table", corner, 1.80 * along_far, 0.95 * towards_near, 0.0025, lit=True),
        Sheet(
            "wrist_floor",
            floor_corner - 2.2 * along_far - 3.2 * towards_near,
            6.5 * along_far,
            5.0 * towards_near,
            0.008,
        ),
        Sheet(
            "wrist_wall_back",
            floor_corner - WALL_BEYOND_FAR_EDGE * towards_near - 2.2 * along_far,
            6.5 * along_far,
            2.7 * up,
            0.006,
        ),
        Sheet(
            "wrist_wall_right",
            floor_corner - PARTITION_BEYOND_RIGHT_EDGE * along_far + 1.6 * towards_near,
            -5.0 * towards_near,
            2.7 * up,
            0.006,
        ),
    ]


def main(args: Args) -> None:
    table = TablePlane.load()
    camera = scene.load_cameras()["right_wrist"]
    sheets = room_sheets(table)
    table_index = next(i for i, sheet in enumerate(sheets) if sheet.name == "wrist_table")
    episodes = episode_data.load(args.dataset_root)
    setups = episode_data.all_setups(args.dataset_root)
    env = YamStationEnv(EnvConfig(cameras=("right_wrist",), scene=scene.SceneOptions(backdrop=False)))
    env.render("right_wrist")
    renderer = env._renderers["right_wrist"]
    model, data = env.model, env.data
    gripper = model.body(camera.parent_body).id
    static_geoms = {model.geom("table").id}
    dynamic = np.array([g not in static_geoms for g in range(model.ngeom)] + [False])  # last entry: "nothing"

    # Pixel rays in the camera's own frame, lens distortion undone once.
    u, v = np.meshgrid(np.arange(camera.width, dtype=np.float64), np.arange(camera.height, dtype=np.float64))
    normalised = cv2.undistortPoints(
        np.stack([u, v], axis=-1).reshape(-1, 1, 2), camera.K, camera.dist_coeffs
    ).reshape(-1, 2)
    rays_cv = np.hstack([normalised, np.ones((len(normalised), 1))])
    rays_cv /= np.linalg.norm(rays_cv, axis=1, keepdims=True)

    def frame_samples(episode, setup, frame: int):  # noqa: ANN001, ANN202
        """For one wrist frame: per sheet, the texel index and colour of every pixel that shows it."""
        pose_scene(env, episode, setup, frame, lag=renderer.image_model.latency)
        seg = renderer.render_segmentation(data).reshape(-1)
        hidden = dynamic[np.where(seg >= 0, seg, model.ngeom)].reshape(camera.height, camera.width)
        image = episode.image("right_wrist", frame)
        masks = real_masks(image)
        hidden = hidden | masks["ball"] | masks["cup_orange"] | masks["cup_blue"] | masks["tips"] | masks["cardboard"]
        hidden = cv2.dilate(hidden.astype(np.uint8), np.ones((13, 13), np.uint8)).reshape(-1) > 0
        rotation, translation = camera.world_to_cv(data.xpos[gripper], data.xmat[gripper].reshape(3, 3))
        origin, rays = -rotation.T @ translation, rays_cv @ rotation
        hits = [sheet.hit(origin, rays) for sheet in sheets]
        nearest = np.argmin(np.stack([h[0] for h in hits]), axis=0)
        seen = np.isfinite(np.min(np.stack([h[0] for h in hits]), axis=0)) & ~hidden
        colours = image.reshape(-1, 3)
        pick = np.flatnonzero(seen & (nearest == table_index))
        rows, cols = sheets[table_index].shape
        index = (hits[table_index][2][pick] * rows).astype(np.int64) * cols + (
            hits[table_index][1][pick] * cols
        ).astype(np.int64)
        sheet_of_pixel = np.where(seen, nearest, -1).reshape(camera.height, camera.width).astype(np.int16)
        return (
            (index, colours[pick]),
            image,
            sheet_of_pixel,
            (data.xpos[gripper].copy(), data.xmat[gripper].reshape(3, 3).copy()),
        )

    jobs = [(name, f) for name, ep in episodes.items() for f in range(0, len(ep), args.every)]
    # The table has no texture to speak of, only a soft sheen, and it is seen in every frame under a
    # different exposure: per texel, keep a grey-level histogram with the colour sum of each bin, and
    # average the most common appearance. The walls and floor carry detail (a poster, chairs), and
    # averaging frames whose poses are each a little off would smear it: there, each texel takes the
    # one frame that saw it best -- nearest the image centre, with the arm at rest. Those are looked
    # up from the texel's side (project the texel into the frame), so neighbouring texels take
    # neighbouring pixels of the same frame and the picture stays coherent.
    n_table = sheets[table_index].shape[0] * sheets[table_index].shape[1]
    bin_counts = np.zeros(n_table * BINS, np.float32)
    bin_sums = np.zeros((n_table * BINS, 3), np.float32)
    best_weight = [np.zeros(s_.shape[0] * s_.shape[1], np.float32) for s_ in sheets]
    best_colour = [np.zeros((s_.shape[0] * s_.shape[1], 3), np.uint8) for s_ in sheets]
    texel_points = []
    for sheet in sheets:
        rows, cols = sheet.shape
        cu, cv_ = np.meshgrid((np.arange(cols) + 0.5) / cols, (np.arange(rows) + 0.5) / rows)
        texel_points.append(sheet.p0 + cu.reshape(-1, 1) * sheet.eu + cv_.reshape(-1, 1) * sheet.ev)
    # The same again per episode (walls and floor only): what that episode's own frames saw of the room,
    # which differed from episode to episode (a chair moved, people walked past).
    own_weight: dict[str, list] = {}
    own_colour: dict[str, list] = {}
    for count, (name, f) in enumerate(jobs):
        episode = episodes[name]
        if name not in own_weight:
            own_weight[name] = [
                None if i == table_index else np.zeros(len(texel_points[i]), np.float32) for i in range(len(sheets))
            ]
            own_colour[name] = [
                None if i == table_index else np.zeros((len(texel_points[i]), 3), np.uint8) for i in range(len(sheets))
            ]
        lo, hi = max(f - 3, 0), min(f + 3, len(episode) - 1)
        still = np.abs(episode.state[hi, :6] - episode.state[lo, :6]).max() < 0.03
        (index, colours), image, sheet_of_pixel, parent = frame_samples(episode, setups[name], f)
        bins = np.minimum(colours.mean(axis=1).astype(np.int64) * BINS // 256, BINS - 1)
        flat = index * BINS + bins
        bin_counts += np.bincount(flat, minlength=len(bin_counts)).astype(np.float32)
        for channel in range(3):
            bin_sums[:, channel] += np.bincount(
                flat, weights=colours[:, channel].astype(np.float64), minlength=len(bin_counts)
            ).astype(np.float32)
        if count % args.detail_every:
            continue
        for i in range(len(sheets)):
            if i == table_index or not (sheet_of_pixel == i).any():
                continue
            projected = camera.project(texel_points[i], *parent)
            px, py = projected[:, 0], projected[:, 1]
            inside = (
                (projected[:, 2] > 0.05) & (px >= 0) & (px <= camera.width - 1) & (py >= 0) & (py <= camera.height - 1)
            )
            candidates = np.flatnonzero(inside)
            ix, iy = np.round(px[candidates]).astype(int), np.round(py[candidates]).astype(int)
            shows_sheet = sheet_of_pixel[iy, ix] == i
            candidates, ix, iy = candidates[shows_sheet], ix[shows_sheet], iy[shows_sheet]
            weight = (
                1.0
                - 2.0 * ((px[candidates] - camera.cx) / camera.width) ** 2
                - 2.0 * ((py[candidates] - camera.cy) / camera.height) ** 2
            ) * (1.0 if still else 0.3)
            sampled = bilinear(image, px[candidates], py[candidates])
            for weights, colours_ in ((best_weight[i], best_colour[i]), (own_weight[name][i], own_colour[name][i])):
                better = weight > weights[candidates]
                weights[candidates[better]] = weight[better]
                colours_[candidates[better]] = sampled[better]

    counts = bin_counts.reshape(n_table, BINS)
    smooth = counts.copy()
    smooth[:, 1:] += 0.5 * counts[:, :-1]
    smooth[:, :-1] += 0.5 * counts[:, 1:]
    mode = smooth.argmax(axis=1)
    member = np.abs(np.arange(BINS)[None, :] - mode[:, None]) <= 1
    table_weight = (counts * member).sum(axis=1)
    table_colour = (bin_sums.reshape(n_table, BINS, 3) * member[..., None]).sum(axis=1) / np.maximum(
        table_weight, 1.0
    )[:, None]

    textures = args.assets / "textures"
    meshes = args.assets / "meshes"
    per_episode = args.assets / "backdrops" / "right_wrist"
    per_episode.mkdir(parents=True, exist_ok=True)
    layout = {"sheets": [], "notes": "generated by calibration/bake_wrist.py from the wrist camera's footage"}
    for i, sheet in enumerate(sheets):
        rows, cols = sheet.shape
        if i == table_index:
            seen = (table_weight >= 3).reshape(rows, cols)
            texture = np.clip(table_colour, 0, 255).reshape(rows, cols, 3).astype(np.uint8)
        else:
            seen = (best_weight[i] > 0).reshape(rows, cols)
            texture = best_colour[i].reshape(rows, cols, 3).copy()
        # Texels no frame ever showed: continue the surface from what is around them.
        fill_value = np.median(texture[seen], axis=0) if seen.any() else np.array([120, 120, 120])
        texture[~seen] = fill_value
        if seen.any() and (~seen).any():
            small = cv2.resize(texture, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            hole = cv2.resize(
                (~seen).astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            filled = cv2.resize(
                cv2.inpaint(small, hole, 5, cv2.INPAINT_TELEA), (cols, rows), interpolation=cv2.INTER_LINEAR
            )
            texture[~seen] = filled[~seen]
        if i == table_index:
            # What is left of frame-to-frame exposure differences is a faint patchwork; the real
            # surface is smooth at this scale, so smooth it, keeping the broad sheen. The unsmoothed
            # texture is kept in the cache for judging how much smoothing that takes.
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(CACHE_DIR / "wrist_table_raw.png"), texture)
            texture = cv2.GaussianBlur(texture, (0, 0), args.table_smoothing / sheet.texel)
        cv2.imwrite(str(textures / f"{sheet.name}.png"), texture)
        if i != table_index:
            # Per episode: its own view of this surface where it had one, the shared texture elsewhere,
            # blended over a few texels. Stored small; the wrist camera never sees the room sharply.
            for name in own_weight:
                own = (own_weight[name][i] > 0).reshape(rows, cols).astype(np.float32)
                own = cv2.GaussianBlur(cv2.erode(own, np.ones((5, 5), np.uint8)), (0, 0), 2.0)[..., None]
                mixed = own_colour[name][i].reshape(rows, cols, 3) * own + texture * (1.0 - own)
                small = cv2.resize(
                    mixed.astype(np.uint8),
                    None,
                    fx=PER_EPISODE_SCALE,
                    fy=PER_EPISODE_SCALE,
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imwrite(str(per_episode / f"{name}_{sheet.name}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
        corners = np.array([sheet.p0, sheet.p0 + sheet.eu, sheet.p0 + sheet.eu + sheet.ev, sheet.p0 + sheet.ev])
        uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        if sheet.normal @ (np.array([0.3, 0.0, 0.3]) - sheet.p0) < 0:  # face the workspace
            faces = faces[:, ::-1]
        save_obj(meshes / f"{sheet.name}.obj", corners, faces, uvs)
        layout["sheets"].append(
            {
                "name": sheet.name,
                "mesh": f"meshes/{sheet.name}.obj",
                "texture": f"textures/{sheet.name}.png",
                "lit": sheet.lit,
            }
        )
        print(f"{sheet.name}: {cols}x{rows} texels, {seen.mean() * 100:.0f}% seen by the wrist camera")
    args.out_layout.write_text(json.dumps(layout, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
