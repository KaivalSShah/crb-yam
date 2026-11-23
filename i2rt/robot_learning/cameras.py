"""USB cameras shared by the viewer and teleop recording."""

import enum
import threading
import time

import cv2
import numpy as np


class CameraRole(enum.Enum):
    OVERHEAD = "overhead"
    RIGHT_WRIST = "right_wrist"


class Orientation(enum.Enum):
    ROTATE_0 = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = 270

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self is Orientation.ROTATE_90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self is Orientation.ROTATE_180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self is Orientation.ROTATE_270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame


class Camera:
    """One capture device. Overhead and wrist cameras are the same class; ``role`` says which."""

    def __init__(self, role: CameraRole, index: int, orientation: Orientation = Orientation.ROTATE_0) -> None:
        self.role = role
        self.index = index
        self.orientation = orientation
        self._capture = cv2.VideoCapture(index)

    def is_open(self) -> bool:
        return bool(self._capture.isOpened())

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        return self.orientation.apply(frame)

    def close(self) -> None:
        self._capture.release()


class LatestFrames:
    """Read cameras on background threads, keeping only each one's newest frame.

    ``Camera.read`` blocks until the next frame arrives (~33 ms at 30 fps), which a control loop
    cannot afford to wait on. ``get`` returns immediately; None until a camera's first frame.
    Does not close the cameras -- their owner does, after ``close``.
    """

    def __init__(self, cameras: dict[CameraRole, Camera]) -> None:
        self._frames: dict[CameraRole, np.ndarray | None] = {role: None for role in cameras}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._run, args=(role, camera), daemon=True, name=f"camera-{role.value}")
            for role, camera in cameras.items()
        ]
        for thread in self._threads:
            thread.start()

    def _run(self, role: CameraRole, camera: Camera) -> None:
        while not self._stop.is_set():
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frames[role] = frame

    def get(self, role: CameraRole) -> np.ndarray | None:
        with self._lock:
            return self._frames.get(role)

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
