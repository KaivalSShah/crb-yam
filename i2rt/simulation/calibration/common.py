"""Helpers shared by the calibration scripts: dataset access, backgrounds, frame selection, FK."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

SIM_ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = SIM_ROOT / "calibration" / "results"
CACHE_DIR = SIM_ROOT / ".cache"
DEFAULT_DATASET = SIM_ROOT.parents[1] / "data" / "ball_pick_and_place"

FINGER_TRAVEL = 0.0475  # metres each linear_4310 finger slides; gripper state 1.0 = fully open


@dataclass
class Episode:
    path: Path
    state: np.ndarray  # (T, 7): six joint angles [rad] + gripper opening in [0, 1]
    action: np.ndarray  # (T, 7): commanded joints + gripper
    timestamp: np.ndarray  # (T,) seconds since the episode started
    meta: dict

    @property
    def name(self) -> str:
        return self.path.name

    def image(self, camera: str, index: int) -> np.ndarray:
        """BGR uint8 frame, exactly as recorded (640x480)."""
        return cv2.imread(str(self.path / camera / f"{index:06d}.jpg"))

    def __len__(self) -> int:
        return len(self.state)


def load_episodes(dataset_root: Path) -> list[Episode]:
    episodes = []
    for path in sorted(Path(dataset_root).glob("episode_*")):
        if not (path / "meta.json").is_file():
            continue
        episodes.append(
            Episode(
                path=path,
                state=np.load(path / "state.npy"),
                action=np.load(path / "action.npy"),
                timestamp=np.load(path / "timestamp.npy"),
                meta=json.loads((path / "meta.json").read_text()),
            )
        )
    if not episodes:
        raise SystemExit(f"no recorded episodes under {dataset_root}")
    return episodes


def modal_background(episodes: list[Episode], camera: str, step: int = 8, cache: bool = True) -> tuple:
    """Per-pixel most common colour over the dataset, and the fraction of frames supporting it.

    A median leaves a ghost wherever the arm spends more than half its time; the mode of a coarse
    grey histogram recovers the table there as long as it is the single most frequent appearance.
    Pixels with low support are ones where nothing static dominates (arm, operator, chair).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = CACHE_DIR / f"background3_{camera}_{len(episodes)}eps_step{step}.npz"
    if cache and key.is_file():
        data = np.load(key)
        return data["background"], data["support"]
    stack = np.stack([ep.image(camera, i) for ep in episodes for i in range(0, len(ep), step)])
    grey = stack.astype(np.float32).mean(axis=3)
    bins = 32
    quant = np.clip((grey / (256 / bins)).astype(np.int16), 0, bins - 1)
    counts = np.stack([(quant == b).sum(0) for b in range(bins)]).astype(np.float32)
    smooth = counts.copy()  # let a cluster straddling a bin edge still win
    smooth[1:] += 0.6 * counts[:-1]
    smooth[:-1] += 0.6 * counts[1:]
    mode = smooth.argmax(0)
    member = np.abs(quant - mode[None]) <= 1
    n = member.sum(0)
    coarse = np.where(member, grey, 0).sum(0) / np.maximum(n, 1)
    # Where no single appearance dominates (a patch the arm shades in half the frames), neighbouring
    # pixels can settle on different modes and the result looks blotchy. There, follow the
    # neighbourhood's choice; pixels with a clear mode keep their own, so thin static details survive.
    ambiguous = (n / len(stack)) < 0.7
    neighbourhood = cv2.medianBlur(coarse.astype(np.float32), 5)
    for _ in range(3):
        neighbourhood = cv2.medianBlur(neighbourhood, 5)
    coarse = np.where(ambiguous, neighbourhood, coarse)
    # Membership by histogram bin leaves visible steps where neighbouring pixels pick neighbouring
    # bins. Re-select around the coarse estimate with a window that does not depend on bin edges.
    member = np.abs(grey - coarse[None]) < 20.0
    n = member.sum(0)
    background = np.where(member[..., None], stack, 0).astype(np.float32).sum(0) / np.maximum(n, 1)[..., None]
    support = (n / len(stack)).astype(np.float32)
    if cache:
        np.savez_compressed(key, background=background.astype(np.float32), support=support)
    return background.astype(np.float32), support


def median_background(episodes: list[Episode], camera: str, step: int = 6, cache: bool = True) -> np.ndarray:
    """Median over episodes of each episode's per-pixel median frame.

    Coarser than ``modal_background`` at removing the arm, but coherent where the scene itself
    changed between episodes (a chair that was moved shows up whole, where the mode is speckled).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = CACHE_DIR / f"median_{camera}_{len(episodes)}eps_step{step}.npy"
    if cache and key.is_file():
        return np.load(key)
    per_episode = []
    for ep in episodes:
        stack = np.stack([ep.image(camera, i) for i in range(0, len(ep), step)])
        per_episode.append(np.median(stack, axis=0))
    background = np.median(np.stack(per_episode), axis=0).astype(np.float32)
    if cache:
        np.save(key, background)
    return background


def robot_model() -> mujoco.MjModel:
    """The bare arm + gripper model (no scene), for forward kinematics and silhouettes."""
    return mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))


def robot_model_with_camera(name: str = "calib", parent_body: str | None = None) -> mujoco.MjModel:
    """The bare robot plus one camera declared with explicit intrinsics, for calibration renders.

    The camera hangs off the world, or off ``parent_body`` (the gripper, for the wrist camera).
    ``PinholeCamera.apply_to_model`` then moves it and sets its intrinsics without recompiling.
    """
    spec = mujoco.MjSpec.from_file(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    parent = spec.worldbody if parent_body is None else spec.body(parent_body)
    cam = parent.add_camera(name=name)
    cam.pos = [0, 0, 1]
    cam.resolution = [640, 480]
    cam.focal_pixel = [500, 500]
    cam.sensor_size = [0.0064, 0.0048]
    return spec.compile()


def set_robot_state(
    model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray, qadr: np.ndarray | None = None
) -> None:
    """Write a recorded 7-vector (6 joints + normalised gripper) into ``data.qpos`` and run FK."""
    if qadr is None:
        qadr = np.array([model.joint(f"joint{i}").qposadr[0] for i in range(1, 9)])
    data.qpos[qadr[:6]] = state[:6]
    data.qpos[qadr[6:]] = float(state[6]) * FINGER_TRAVEL
    mujoco.mj_forward(model, data)


def select_static_frames(episodes: list[Episode], count: int, max_speed: float = 0.02, seed: int = 0) -> list:
    """(episode index, frame index) pairs where the arm is nearly still, spread over the workspace.

    Stillness matters because a camera frame and the joint state logged with it are not captured at
    the same instant; diversity (farthest-point sampling of the grasp point) conditions the fit.
    """
    model = robot_model()
    data = mujoco.MjData(model)
    site = model.site("grasp_site").id
    candidates, positions = [], []
    for e, ep in enumerate(episodes):
        s = ep.state
        if len(s) < 12:
            continue
        speed = np.abs(s[6:, :6] - s[:-6, :6]).max(axis=1)  # change over 6 frames (0.2 s)
        for t in range(3, len(s) - 3, 3):
            if speed[t - 3] < max_speed:
                set_robot_state(model, data, s[t])
                candidates.append((e, t))
                positions.append(np.concatenate([data.site_xpos[site], 0.15 * data.site_xmat[site][[2, 5, 8]]]))
    positions = np.array(positions)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(candidates)))]
    dist = np.linalg.norm(positions - positions[chosen[0]], axis=1)
    while len(chosen) < min(count, len(candidates)):
        nxt = int(dist.argmax())
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(positions - positions[nxt], axis=1))
    return [candidates[i] for i in chosen]
