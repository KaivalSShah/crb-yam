"""Render the simulated cameras the way the real ones deliver frames: 640x480, lens and all.

MuJoCo draws an ideal pinhole image. Each real camera adds a lens (radial distortion), a sensor
and ISP (blur, noise, exposure, colour) and the recorder's resize to 640x480. ``CameraRenderer``
reproduces that chain: it renders the pinhole image at a higher resolution on a canvas wide enough
for the lens model, warps it through the lens, and area-averages it down like the recorder does.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from i2rt.simulation import scene
from i2rt.simulation.cameras import PinholeCamera

# Which backdrop a camera sees: each real camera photographed the room itself, and the photographs
# only line up from that camera's own viewpoint.
_BACKDROP_GROUP = {"overhead": scene.GROUP_OVERHEAD_ONLY, "right_wrist": scene.GROUP_WRIST_ONLY}


@dataclass
class ImageModel:
    """What a camera's sensor and image pipeline do to the picture the lens forms.

    Values per camera are measured from the recorded footage (``calibration/results/image_models.json``;
    the numbers and how each was measured are in the README).
    """

    blur_sigma: float = 0.0
    """Softness in pixels of the recorded image: lens, demosaicing and JPEG together."""
    sharpen: float = 0.0
    """Strength of the camera's edge enhancement (unsharp mask, 1 px radius). Webcams sharpen their
    output; without it a render is softer than the footage."""
    noise_sigma: float = 0.0
    """Per-pixel noise in grey levels."""
    auto_exposure: float = 0.0
    """How strongly the camera pulls each frame's mean brightness toward ``exposure_target``:
    0 leaves the frame alone, 1 pins its mean. A wrist camera that swings from the dark table to a
    white wall re-exposes as it goes; this is that."""
    exposure_target: float = 100.0
    """Mean grey level the auto-exposure aims for."""
    flicker: float = 0.0
    """Frame-to-frame random brightness variation (standard deviation of the gain), from mains-lit
    scenes seen through a short global shutter."""
    jpeg_quality: int = 0
    """Pass each frame through JPEG at this quality, as the recorder does when it saves it (0: off)."""
    exposure_gain: float = 1.0
    """Overall brightness factor on top of everything else. 1 as calibrated; the environment's
    appearance randomisation varies it."""
    latency: float = 0.0
    """Seconds by which a frame is older than the joint state delivered with it (exposure, USB, decode).
    Applied by the environment, which hands out frames rendered that much earlier."""

    @classmethod
    def for_camera(cls, name: str) -> ImageModel:
        return cls(**scene.load_json("image_models.json", default={}).get(name, {}))


class CameraRenderer:
    """Renders one calibrated camera of a scene model."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera: PinholeCamera,
        supersample: int = 2,
        image_model: ImageModel | None = None,
        seed: int = 0,
        soft_shadows: bool = True,
    ) -> None:
        self.model = model
        self.camera = camera
        self.image_model = image_model or ImageModel.for_camera(camera.name)
        self.cam_id = model.camera(camera.name).id
        distorted = bool(camera.k1 or camera.k2)
        self.canvas = _canvas_for(camera) if distorted else 1.0
        self.render_w = round(camera.width * self.canvas * supersample)
        self.render_h = round(camera.height * self.canvas * supersample)
        camera.apply_to_model(model, self.cam_id, canvas_scale=self.canvas)
        self._renderer = mujoco.Renderer(model, self.render_h, self.render_w)
        self._maps = (
            camera.distortion_maps(self.render_w, self.render_h, self.canvas, out_scale=supersample)
            if distorted
            else None
        )
        self._option = mujoco.MjvOption()
        self._option.geomgroup[:] = 0
        for group in (0, 1, scene.GROUP_VISUAL, _BACKDROP_GROUP.get(camera.name, scene.GROUP_OVERHEAD_ONLY)):
            self._option.geomgroup[group] = 1
        self._option.sitegroup[:] = 0
        self._rng = np.random.default_rng(seed)
        self._segmenter: mujoco.Renderer | None = None
        self.soft_shadows = soft_shadows
        lighting = scene.lighting_parameters()
        self.shadow_blur = lighting["shadow_blur"]  # px of penumbra at the recorded resolution
        self.shadow_strength = lighting["shadow_strength"]
        self._shadow_renderer = (
            mujoco.Renderer(model, max(self.render_h // 4, 120), max(self.render_w // 4, 160))
            if soft_shadows
            else None
        )
        self.sheen = TableSheen.for_camera(model, camera)
        # Each real camera has its own colour response (the cup the overhead camera sees as orange is
        # yellow to the wrist camera), so paints are calibrated per camera and swapped in before drawing.
        colours = scene.load_json("appearance.json", default={}).get(camera.name, {})
        self._material_ids = np.array(
            [model.material(name).id for name in colours if _has_material(model, name)], dtype=int
        )
        self._material_rgba = np.array(
            [colours[name] for name in colours if _has_material(model, name)], dtype=float
        ).reshape(-1, 4)
        self._calibrated_rgba = self._material_rgba.copy()

    def _use_camera_colours(self) -> None:
        if len(self._material_ids):
            self.model.mat_rgba[self._material_ids] = self._material_rgba

    def render(self, data: mujoco.MjData, effects: bool = True, noise: bool = True) -> np.ndarray:
        """RGB uint8 image (height, width, 3) of the scene in ``data``, as the real camera would record it.

        ``effects`` applies the camera's image model; ``noise`` its random parts (sensor noise and
        flicker), which calibration turns off to compare like with like.
        """
        self._use_camera_colours()
        self._renderer.update_scene(data, camera=self.cam_id, scene_option=self._option)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        image = self._renderer.render()
        if self.soft_shadows:
            image = self._shade(image, data)
        if self._maps is not None:
            image = cv2.remap(image, self._maps[0], self._maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        image = cv2.resize(image, (self.camera.width, self.camera.height), interpolation=cv2.INTER_AREA)
        if self.sheen is not None:
            image = self.sheen.apply(image, data, self._coarse_segmentation(data))
        return self._apply_image_model(image, noise) if effects else image

    def _shade(self, image: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        """Darken ``image`` with soft shadows.

        MuJoCo's shadow maps have hard, blocky edges, nothing like the diffuse shadows office ceiling
        lights cast. So the picture itself is drawn unshadowed, and shadow is added as a separate
        layer: a small render with and without shadows gives how much each pixel is darkened, that
        ratio is blurred to a penumbra, and the picture is multiplied by it.
        """
        self._shadow_renderer.update_scene(data, camera=self.cam_id, scene_option=self._option)
        self._shadow_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        shadowed = self._shadow_renderer.render().astype(np.float32).mean(axis=2)
        self._shadow_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        lit = self._shadow_renderer.render().astype(np.float32).mean(axis=2)
        ratio = np.clip((shadowed + 4.0) / (lit + 4.0), 0.0, 1.0)
        ratio = cv2.GaussianBlur(ratio, (0, 0), self.shadow_blur * ratio.shape[1] / self.camera.width / self.canvas)
        ratio = 1.0 - self.shadow_strength * (1.0 - ratio)
        ratio = cv2.resize(ratio, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return np.clip(image.astype(np.float32) * ratio[..., None], 0, 255).astype(np.uint8)

    def _coarse_segmentation(self, data: mujoco.MjData) -> np.ndarray:
        """Geom id per pixel at a fraction of the recorded size: enough to tell where the table is."""
        small = self._shadow_renderer
        if small is None or (self._maps is not None and self.canvas > 1.1):
            return self.render_segmentation(data, size=(160, 120))
        small.update_scene(data, camera=self.cam_id, scene_option=self._option)
        small.enable_segmentation_rendering()
        seg = small.render()
        small.disable_segmentation_rendering()
        ids = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        # The render covers the lens canvas; crop to the part the camera actually records.
        margin_y = round(ids.shape[0] * (1 - 1 / self.canvas) / 2)
        margin_x = round(ids.shape[1] * (1 - 1 / self.canvas) / 2)
        return ids[margin_y : ids.shape[0] - margin_y, margin_x : ids.shape[1] - margin_x].astype(np.int32)

    def randomize(
        self,
        rng: np.random.Generator | None,
        shift: float = 0.004,
        tilt: float = 0.01,
        colour: float = 0.06,
        exposure: float = 0.06,
    ) -> None:
        """Perturb this camera's pose, paints and exposure a little; ``rng=None`` restores the calibration.

        ``shift`` [m] and ``tilt`` [rad] bound the camera's displacement and rotation, ``colour`` the
        relative change of each paint channel, ``exposure`` that of overall brightness.
        """
        camera = PinholeCamera(**{**self.camera.__dict__})
        self._material_rgba = self._calibrated_rgba.copy()
        self.image_model.exposure_gain = 1.0
        if rng is not None:
            camera.pos = (np.asarray(camera.pos) + rng.uniform(-shift, shift, 3)).tolist()
            twist = np.zeros(4)
            axis = rng.normal(size=3)
            mujoco.mju_axisAngle2Quat(twist, axis / np.linalg.norm(axis), rng.uniform(-tilt, tilt))
            turned = np.zeros(4)
            mujoco.mju_mulQuat(turned, np.asarray(camera.quat, dtype=float), twist)
            camera.quat = turned.tolist()
            self._material_rgba[:, :3] *= 1.0 + rng.uniform(-colour, colour, self._material_rgba[:, :3].shape)
            self.image_model.exposure_gain = float(1.0 + rng.uniform(-exposure, exposure))
        camera.apply_to_model(self.model, self.cam_id, canvas_scale=self.canvas)

    def set_texture(self, name: str, image: np.ndarray) -> None:
        """Replace a texture's pixels (RGB, any size) in the model and in this camera's GL contexts."""
        texture = self.model.texture(name).id
        height, width = int(self.model.tex_height[texture]), int(self.model.tex_width[texture])
        start = int(self.model.tex_adr[texture])
        pixels = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        self.model.tex_data[start : start + height * width * 3] = np.ascontiguousarray(pixels).reshape(-1)
        for renderer in (self._renderer, self._shadow_renderer):
            if renderer is not None:
                renderer._gl_context.make_current()
                mujoco.mjr_uploadTexture(self.model, renderer._mjr_context, texture)

    def render_segmentation(self, data: mujoco.MjData, size: tuple[int, int] | None = None) -> np.ndarray:
        """Geom id per pixel (-1 where nothing is drawn), at the recorded size or at ``size`` (width, height)."""
        if self._segmenter is None:
            self._segmenter = mujoco.Renderer(self.model, self.render_h, self.render_w)
            self._segmenter.enable_segmentation_rendering()
        self._segmenter.update_scene(data, camera=self.cam_id, scene_option=self._option)
        seg = self._segmenter.render()
        ids = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1).astype(np.float32)
        if self._maps is not None:
            ids = cv2.remap(ids, self._maps[0], self._maps[1], cv2.INTER_NEAREST, borderValue=-1)
        target = size or (self.camera.width, self.camera.height)
        return cv2.resize(ids, target, interpolation=cv2.INTER_NEAREST).astype(np.int32)

    def _apply_image_model(self, image: np.ndarray, noise: bool) -> np.ndarray:
        im = self.image_model
        out = image.astype(np.float32)
        gain = 1.0
        if im.auto_exposure > 0:
            gain *= (im.exposure_target / max(float(out.mean()), 1.0)) ** im.auto_exposure
        if noise and im.flicker > 0:
            gain *= 1.0 + im.flicker * float(self._rng.standard_normal())
        gain *= im.exposure_gain
        if gain != 1.0:
            out *= gain
        if im.blur_sigma > 0:
            out = cv2.GaussianBlur(out, (0, 0), im.blur_sigma)
        if im.sharpen > 0:
            out += im.sharpen * (out - cv2.GaussianBlur(out, (0, 0), 1.0))
        if noise and im.noise_sigma > 0:
            out += self._rng.normal(0.0, im.noise_sigma, out.shape).astype(np.float32)
        frame = np.clip(out, 0, 255).astype(np.uint8)
        if im.jpeg_quality > 0:  # as the recorder saves it: a BGR JPEG
            encoded = cv2.imencode(".jpg", frame[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, im.jpeg_quality])[1]
            frame = np.ascontiguousarray(cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1])
        return frame

    def close(self) -> None:
        self._renderer.close()
        for extra in (self._segmenter, self._shadow_renderer):
            if extra is not None:
                extra.close()


class TableSheen:
    """The glossy table's view-dependent highlight, added to a camera's table pixels.

    ``calibration/sheen.py`` measured, from the real footage, how much brighter or darker than its
    photograph the table looks as a function of the direction a line of sight leaves in after
    mirroring off it -- a blurred image of the ceiling lights. Here each pixel's mirror direction is
    looked up in that map and the difference added wherever the pixel shows the table.
    """

    GRID = (64, 48)  # the highlight is smooth: evaluate it on a coarse grid of pixels and scale up

    def __init__(self, model: mujoco.MjModel, camera: PinholeCamera, sheen_map: np.ndarray, table_geom: str) -> None:
        self.map = sheen_map
        self.table_geom = model.geom(table_geom).id
        self.cam_id = model.camera(camera.name).id
        self.normal = scene.TablePlane.load().normal
        gx, gy = np.meshgrid(
            (np.arange(self.GRID[0]) + 0.5) * camera.width / self.GRID[0] - 0.5,
            (np.arange(self.GRID[1]) + 0.5) * camera.height / self.GRID[1] - 0.5,
        )
        pixels = np.stack([gx, gy], axis=-1).reshape(-1, 1, 2)
        normalised = cv2.undistortPoints(pixels, camera.K, camera.dist_coeffs).reshape(-1, 2)
        rays = np.hstack([normalised, np.ones((len(normalised), 1))])
        self._rays_cv = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        u = np.cross([0.0, 1.0, 0.0], self.normal)
        self._u = u / np.linalg.norm(u)
        self._v = np.cross(self.normal, self._u)

    @classmethod
    def for_camera(cls, model: mujoco.MjModel, camera: PinholeCamera) -> TableSheen | None:
        stem = camera.name.removeprefix("right_")
        path = scene.ASSETS / f"{stem}_table_sheen.npy"
        if not path.is_file() or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{stem}_table") < 0:
            return None
        return cls(model, camera, np.load(path), f"{stem}_table")

    def apply(self, image: np.ndarray, data: mujoco.MjData, segmentation: np.ndarray) -> np.ndarray:
        # MuJoCo's camera frame looks down -Z with +Y up; the rays above are in the OpenCV convention.
        camera_to_world = data.cam_xmat[self.cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        rays = self._rays_cv @ camera_to_world.T
        reflected = rays - 2.0 * (rays @ self.normal)[:, None] * self.normal
        scale = 0.5 * (self.map.shape[0] - 1)
        map_x = ((reflected @ self._u + 1.0) * scale).astype(np.float32).reshape(self.GRID[1], self.GRID[0])
        map_y = ((reflected @ self._v + 1.0) * scale).astype(np.float32).reshape(self.GRID[1], self.GRID[0])
        sheen = cv2.remap(self.map, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        size = (image.shape[1], image.shape[0])
        sheen = cv2.resize(sheen, size, interpolation=cv2.INTER_CUBIC)
        on_table = cv2.resize(
            (segmentation == self.table_geom).astype(np.float32), size, interpolation=cv2.INTER_LINEAR
        )
        on_table = cv2.GaussianBlur(on_table, (0, 0), 2.0)
        return np.clip(image.astype(np.float32) + sheen * on_table[..., None], 0, 255).astype(np.uint8)


def _canvas_for(camera: PinholeCamera) -> float:
    """How much wider than the recorded image the pinhole render must be for this lens.

    A barrel-distorting lens records, at its borders, light from outside the pinhole frame of the same
    focal length. Undistorting the image border says how far outside.
    """
    xs = np.linspace(0, camera.width - 1, 9)
    ys = np.linspace(0, camera.height - 1, 9)
    border = np.array(
        [[x, y] for x in xs for y in (0, camera.height - 1)] + [[x, y] for y in ys for x in (0, camera.width - 1)]
    )
    ideal = cv2.undistortPoints(border.reshape(-1, 1, 2).astype(np.float64), camera.K, camera.dist_coeffs).reshape(
        -1, 2
    )
    reach_x = np.abs(ideal[:, 0] * camera.fx).max() / max(camera.cx, camera.width - 1 - camera.cx)
    reach_y = np.abs(ideal[:, 1] * camera.fy).max() / max(camera.cy, camera.height - 1 - camera.cy)
    return float(max(1.0, reach_x, reach_y) * 1.03 + 0.02)


def _has_material(model: mujoco.MjModel, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name) >= 0


def make_renderers(model: mujoco.MjModel, supersample: int = 2) -> dict[str, CameraRenderer]:
    """One renderer per calibrated camera, keyed by the dataset's camera names."""
    return {name: CameraRenderer(model, camera, supersample) for name, camera in scene.load_cameras().items()}
