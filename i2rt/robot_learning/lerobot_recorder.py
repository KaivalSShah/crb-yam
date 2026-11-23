"""Record leader-follower teleop as a LeRobot dataset.

Each saved episode is one stretch of following, after the ease-in has finished.
``observation.state`` is the follower's measured joints, ``action`` is the joint
command sent to the follower, and the two cameras are ``observation.images.overhead``
and ``observation.images.right_wrist``. Install LeRobot before recording:

    uv pip install lerobot
"""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


def _recorded_episodes(root: Path) -> int | None:
    """Episodes a dataset at ``root`` holds per its info.json; None when there is no dataset there."""
    info = root / "meta" / "info.json"
    if not info.is_file():
        return None
    try:
        return int(json.loads(info.read_text()).get("total_episodes", 0))
    except (ValueError, OSError):
        return None  # unreadable: leave it alone rather than guess


def joint_names(num_dofs: int, n_arm: int) -> list[str]:
    names = [f"joint_{i + 1}" for i in range(n_arm)]
    if num_dofs > n_arm:
        names.append("gripper")
    return names


def teleop_features(names: list[str]) -> dict[str, dict[str, Any]]:
    """Feature schema ACT and the other LeRobot policies read back."""
    joints = {"dtype": "float32", "shape": (len(names),), "names": names}
    image = {
        "dtype": "video",
        "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        "observation.state": joints,
        "action": joints,
        "observation.images.overhead": image,
        "observation.images.right_wrist": image,
    }


def _rgb(frame: np.ndarray) -> np.ndarray:
    import cv2

    resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


class TeleopDataset:
    """Append teleop episodes to a local LeRobot dataset, creating it on first use.

    ``save_episode`` encodes that episode's video on the spot: ~1 s per second of footage, during
    which another thread was held up by as much as 205 ms (bench, 2026-11-23; the motors fault at
    400 ms). Deferring it with ``batch_encoding_size`` is not an option on lerobot 0.4.x, whose batch
    encoder reads episode metadata that has not been flushed yet and fails.
    """

    def __init__(self, root: Path, repo_id: str, fps: int, names: list[str], task: str, robot_type: str) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._task = task
        self._root = root
        if _recorded_episodes(root) == 0:
            # Created, but no episode was ever saved: lerobot 0.4.x cannot load it (meta/tasks.parquet
            # appears with the first episode) and falls back to looking it up on the Hugging Face hub.
            # It holds no recordings, so start it over.
            shutil.rmtree(root)
            print(f"{root} had no recorded episodes; starting it over", flush=True)
        if (root / "meta" / "info.json").is_file():
            if hasattr(LeRobotDataset, "resume"):
                self._dataset = LeRobotDataset.resume(repo_id=repo_id, root=root, streaming_encoding=False)
            else:
                # lerobot 0.4.x has no resume(): load the local dataset and restart its image writer,
                # which create() would otherwise have started.
                self._dataset = LeRobotDataset(repo_id, root=root)
                self._dataset.start_image_writer(num_threads=4)
            print(f"Resuming dataset at {root}", flush=True)
        else:
            # LeRobotDataset.create makes the root itself and refuses one that already exists, so only
            # the parent is created here. An empty root (left by an earlier failed start) is cleared;
            # anything else there is not a dataset we can resume, and is left alone.
            root.parent.mkdir(parents=True, exist_ok=True)
            if root.is_dir() and not any(root.iterdir()):
                root.rmdir()
            elif root.exists():
                raise SystemExit(f"{root} exists but is not a LeRobot dataset; pick another --dataset-root")
            self._dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=teleop_features(names),
                root=root,
                robot_type=robot_type,
                use_videos=True,
                image_writer_threads=4,
            )
            print(f"Recording a new dataset at {root}", flush=True)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        overhead: np.ndarray,
        wrist: np.ndarray,
        task: str | None = None,
    ) -> None:
        """Add one frame; ``task`` overrides the dataset's default label for it."""
        self._dataset.add_frame(
            {
                "observation.state": np.asarray(state, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "observation.images.overhead": _rgb(overhead),
                "observation.images.right_wrist": _rgb(wrist),
                "task": self._task if task is None else task,
            }
        )

    def _has_pending_frames(self) -> bool:
        """Frames added since the last saved episode. lerobot 0.4.x lacks the method; read its buffer."""
        if hasattr(self._dataset, "has_pending_frames"):
            return bool(self._dataset.has_pending_frames())
        buffer = getattr(self._dataset, "episode_buffer", None)
        return buffer is not None and buffer.get("size", 0) > 0

    def save_episode(self) -> None:
        if not self._has_pending_frames():
            return
        # Process pools for video encoding are unreliable on macOS.
        self._dataset.save_episode(parallel_encoding=False)
        print(f"Saved episode to {self._root}", flush=True)

    def close(self) -> None:
        self.save_episode()
        self._dataset.finalize()
