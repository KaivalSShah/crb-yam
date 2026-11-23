"""Show the overhead and right-wrist cameras side by side. Press q to quit.

A missing index is skipped. On a Mac, index 0 is often the built-in camera.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import tyro

from i2rt.robot_learning.cameras import Camera, CameraRole, Orientation


@dataclass
class Args:
    overhead_index: int = 1
    """Device index of the overhead camera."""
    overhead_orientation: Orientation = Orientation.ROTATE_0
    right_wrist_index: int = 0
    """Device index of the right wrist camera."""
    right_wrist_orientation: Orientation = Orientation.ROTATE_180


def _open_cameras(args: Args) -> list[Camera]:
    cameras = [
        Camera(CameraRole.OVERHEAD, args.overhead_index, args.overhead_orientation),
        Camera(CameraRole.RIGHT_WRIST, args.right_wrist_index, args.right_wrist_orientation),
    ]
    opened: list[Camera] = []
    for camera in cameras:
        if camera.is_open():
            opened.append(camera)
            print(f"{camera.role.value} open on index {camera.index}", flush=True)
        else:
            camera.close()
            print(f"{camera.role.value} not available on index {camera.index}", flush=True)
    if not opened:
        raise SystemExit("no cameras opened")
    return opened


def _fit(frame: np.ndarray, panel_width: int, panel_height: int) -> np.ndarray:
    """Scale ``frame`` to fit inside the panel and center it. Empty space stays black."""
    height, width = frame.shape[:2]
    scale = min(panel_width / width, panel_height / height)
    fitted_width = max(1, round(width * scale))
    fitted_height = max(1, round(height * scale))
    resized = cv2.resize(frame, (fitted_width, fitted_height), interpolation=cv2.INTER_AREA)
    panel = np.zeros((panel_height, panel_width, frame.shape[2]), dtype=frame.dtype)
    y = (panel_height - fitted_height) // 2
    x = (panel_width - fitted_width) // 2
    panel[y : y + fitted_height, x : x + fitted_width] = resized
    return panel


def split_view(overhead: np.ndarray | None, wrist: np.ndarray | None) -> np.ndarray:
    """Overhead on the left, right wrist on the right, each in an equal panel."""
    frames = [frame for frame in (overhead, wrist) if frame is not None]
    panel_height = 720
    if frames:
        widest = max(frame.shape[1] / frame.shape[0] for frame in frames)
        panel_width = round(panel_height * widest)
    else:
        panel_width = 1280
    blank = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    left = _fit(overhead, panel_width, panel_height) if overhead is not None else blank
    right = _fit(wrist, panel_width, panel_height) if wrist is not None else blank
    return np.hstack((left, right))


WINDOW = "cameras"


def show_cameras(overhead: np.ndarray | None, wrist: np.ndarray | None, window: str = WINDOW) -> int:
    """Draw the split view and let the window process events. Returns the key pressed, or -1.

    Call it from the main thread: macOS only draws OpenCV windows from there.
    """
    cv2.imshow(window, split_view(overhead, wrist))
    key = cv2.waitKey(1)
    return -1 if key < 0 else key & 0xFF


def close_camera_window() -> None:
    # No waitKey here: on macOS it hung at exit after a run whose arm had failed (bench, 2026-11-23).
    cv2.destroyAllWindows()


def read_cameras(args: Args) -> None:
    cameras = {camera.role: camera for camera in _open_cameras(args)}
    try:
        while True:
            overhead = cameras[CameraRole.OVERHEAD].read() if CameraRole.OVERHEAD in cameras else None
            wrist = cameras[CameraRole.RIGHT_WRIST].read() if CameraRole.RIGHT_WRIST in cameras else None
            if show_cameras(overhead, wrist) == ord("q"):
                break
    finally:
        for camera in cameras.values():
            camera.close()
        close_camera_window()


if __name__ == "__main__":
    read_cameras(tyro.cli(Args))
