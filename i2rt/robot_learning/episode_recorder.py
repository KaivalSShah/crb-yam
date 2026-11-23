"""Record teleop demonstrations as plain episode folders, with no conversion while recording.

Each stretch of following becomes one folder under the demo root::

    <demo root>/episode_0000/
        overhead/000000.jpg ...      camera frames, 640x480 BGR JPEGs
        right_wrist/000000.jpg ...
        state.npy                    follower's measured joints, one row per frame
        action.npy                   joint command sent to the follower, one row per frame
        timestamp.npy                seconds since the episode started
        meta.json                    fps, task, joint names, robot type, cameras, frame count

``meta.json`` is written last, by the same thread that writes the frames, so a folder without it
was cut off mid-episode and is skipped by ``convert_to_lerobot``. Encoding into a LeRobot dataset
is left to that separate step: doing it while recording took ~1-2 s per second of footage and kept
the teleop loop from starting the next episode.
"""

import json
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERAS = ("overhead", "right_wrist")
JPEG_QUALITY = 95


def episode_dirs(root: Path) -> list[Path]:
    """Complete episodes under ``root``, in recording order."""
    return sorted(p for p in root.glob("episode_*") if (p / "meta.json").is_file())


class EpisodeRecorder:
    """Write episodes as they are recorded. ``add`` never waits on the disk; a thread does the writing."""

    def __init__(self, root: Path, fps: int, task: str, joint_names: list[str], robot_type: str) -> None:
        self._root = root
        self._meta = {
            "fps": fps,
            "task": task,
            "joint_names": joint_names,
            "robot_type": robot_type,
            "cameras": list(CAMERAS),
            "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        }
        root.mkdir(parents=True, exist_ok=True)
        existing = [int(p.name.split("_")[1]) for p in root.glob("episode_*") if p.name.split("_")[1].isdigit()]
        self._next = max(existing, default=-1) + 1
        self._current: Path | None = None
        self._start = 0.0
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._times: list[float] = []
        self._failed: str | None = None
        self._session: list[Path] = []  # episodes completed this session, oldest first
        self._writes: queue.Queue = queue.Queue()
        self._writer = threading.Thread(target=self._write, daemon=True, name="episode-writer")
        self._writer.start()
        print(f"Recording episodes to {root} ({len(episode_dirs(root))} already there).", flush=True)

    @property
    def failed(self) -> str | None:
        """Why recording stopped, or None while it works. Once set, nothing more is recorded."""
        return self._failed

    def _fail(self, error: Exception) -> None:
        # Never raises: an exception here would end the teleop loop and leave both arms limp, which
        # is worse than losing the recording. The caller sees ``failed`` and stops following instead.
        if self._failed is None:
            self._failed = f"{type(error).__name__}: {error}"
            print(f"\n*** RECORDING STOPPED -- {self._failed}\n*** Nothing more is saved this session.", flush=True)

    def add(self, state: np.ndarray, action: np.ndarray, overhead: np.ndarray, wrist: np.ndarray) -> None:
        if self._failed is not None:
            return
        if self._current is None:
            episode = self._root / f"episode_{self._next:04d}"
            try:
                for camera in CAMERAS:
                    (episode / camera).mkdir(parents=True)
            except OSError as e:
                self._fail(e)
                return
            self._current = episode  # only once its folders exist
            self._next += 1
            self._start = time.monotonic()
        i = len(self._states)
        for camera, frame in zip(CAMERAS, (overhead, wrist), strict=True):
            small = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
            self._writes.put(("image", self._current / camera / f"{i:06d}.jpg", small))
        self._states.append(np.asarray(state, dtype=np.float32).copy())
        self._actions.append(np.asarray(action, dtype=np.float32).copy())
        self._times.append(time.monotonic() - self._start)

    def end_episode(self) -> None:
        """Close the current episode, if it has frames. Returns at once; the writer finishes it."""
        if self._current is None or not self._states or self._failed is not None:
            # Nothing recorded, or a write failed: left without meta.json, convert_to_lerobot skips it.
            self._current, self._states, self._actions, self._times = None, [], [], []
            return
        meta = {**self._meta, "frames": len(self._states)}
        self._writes.put(
            ("finish", self._current, np.stack(self._states), np.stack(self._actions), np.array(self._times), meta)
        )
        print(f"Saved {self._current.name} ({len(self._states)} frames).", flush=True)
        self._session.append(self._current)
        self._current, self._states, self._actions, self._times = None, [], [], []

    @property
    def recording_episode(self) -> bool:
        """Whether an episode is in progress (frames added since the last ``end_episode``)."""
        return self._current is not None

    def delete_last_episode(self) -> str | None:
        """Delete the most recent episode completed this session; returns its name, or None.

        Refuses mid-episode and never reaches episodes from earlier sessions, so a stray press cannot
        eat older data. The writer thread does the deleting, after that episode's last frames, so this
        returns at once. Numbers are not reused: the next episode must not collide with the delete.
        """
        if self.recording_episode or not self._session:
            return None
        episode = self._session.pop()
        self._writes.put(("delete", episode))
        return episode.name

    @property
    def deletable(self) -> int:
        """Episodes from this session that ``delete_last_episode`` can still remove."""
        return len(self._session)

    def _write(self) -> None:
        while True:
            item = self._writes.get()
            try:
                if item is None:
                    return
                if item[0] == "delete":
                    shutil.rmtree(item[1], ignore_errors=True)  # still honoured after a failure
                    continue
                if self._failed is not None:
                    continue  # drain the queue; nothing more is written
                if item[0] == "image":
                    # imwrite reports failure by returning False (e.g. disk full), not by raising.
                    if not cv2.imwrite(str(item[1]), item[2], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                        raise OSError(f"could not write {item[1]}")
                else:
                    _, episode, states, actions, times, meta = item
                    np.save(episode / "state.npy", states)
                    np.save(episode / "action.npy", actions)
                    np.save(episode / "timestamp.npy", times)
                    (episode / "meta.json").write_text(json.dumps(meta, indent=2))  # last: marks it complete
            except Exception as e:  # any write error stops recording, never the arms
                self._fail(e)
            finally:
                self._writes.task_done()

    def close(self) -> None:
        """End the current episode and wait for every frame to reach the disk."""
        self.end_episode()
        self._writes.put(None)
        self._writer.join()
