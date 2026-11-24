"""Small triangle-mesh helpers (numpy only): STL I/O, plane cuts, primitive solids.

The robot ships as one STL per link, but the real arm is two-tone (white link shells, black motor
housings), so links have to be cut into separately coloured pieces. These helpers do that without
pulling in a mesh library.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np


def load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary STL and weld identical vertices. Returns (vertices Nx3, faces Mx3)."""
    raw = Path(path).read_bytes()
    count = struct.unpack("<I", raw[80:84])[0]
    if len(raw) < 84 + 50 * count:
        raise ValueError(f"{path} is not a binary STL")
    record = np.dtype([("normal", "<f4", 3), ("tri", "<f4", (3, 3)), ("attr", "<u2")])
    tris = np.frombuffer(raw, dtype=record, count=count, offset=84)["tri"].astype(np.float64)
    return weld(tris.reshape(-1, 3), np.arange(3 * count).reshape(-1, 3))


def save_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    tris = vertices[faces].astype("<f4")
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    record = np.zeros(len(tris), dtype=[("normal", "<f4", 3), ("tri", "<f4", (3, 3)), ("attr", "<u2")])
    record["normal"], record["tri"] = normals, tris
    # Written to a private name and moved into place: several processes (parallel tests, calibration
    # workers) may build the same cached mesh at once, and none must ever read a half-written file.
    scratch = Path(path).with_name(f".{Path(path).name}.{os.getpid()}")
    scratch.write_bytes(b"\0" * 80 + struct.pack("<I", len(tris)) + record.tobytes())
    os.replace(scratch, path)


def weld(vertices: np.ndarray, faces: np.ndarray, decimals: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices and drop degenerate faces."""
    _, index, inverse = np.unique(np.round(vertices, decimals), axis=0, return_index=True, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return vertices[index], faces[keep]


def compact(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop vertices no face refers to."""
    used = np.unique(faces)
    remap = np.full(len(vertices), -1)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def split_by_plane(
    vertices: np.ndarray, faces: np.ndarray, point: np.ndarray, normal: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Cut a mesh by a plane. Returns (part on the +normal side, part on the -normal side).

    Triangles straddling the plane are clipped exactly, so the two parts meet along the plane with
    no gap. The cut is left open (no cap): the parts are always drawn together.
    """
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    dist = ((vertices - np.asarray(point, dtype=float)) * normal).sum(axis=1)
    tri = vertices[faces]
    d = dist[faces]
    above, below = (d >= 0).all(axis=1), (d <= 0).all(axis=1)
    pos_tris, neg_tris = [tri[above]], [tri[below & ~above]]
    for t, dd in zip(tri[~above & ~below], d[~above & ~below], strict=True):
        # Walk the triangle's edges, emitting each vertex and each crossing to the side(s) it is on.
        pos_poly, neg_poly = [], []
        for i in range(3):
            a, b, da, db = t[i], t[(i + 1) % 3], dd[i], dd[(i + 1) % 3]
            if da >= 0:
                pos_poly.append(a)
            if da <= 0:
                neg_poly.append(a)
            if (da > 0 and db < 0) or (da < 0 and db > 0):
                cross = a + (b - a) * (da / (da - db))
                pos_poly.append(cross)
                neg_poly.append(cross)
        for poly, out in ((pos_poly, pos_tris), (neg_poly, neg_tris)):
            if len(poly) >= 3:
                out.append(np.array([[poly[0], poly[k], poly[k + 1]] for k in range(1, len(poly) - 1)]))
    parts = []
    for tris in (pos_tris, neg_tris):
        soup = np.concatenate([t.reshape(-1, 3, 3) for t in tris]) if tris else np.zeros((0, 3, 3))
        parts.append(weld(soup.reshape(-1, 3), np.arange(3 * len(soup)).reshape(-1, 3)))
    return parts[0], parts[1]


def select_faces(vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sub-mesh made of the faces where ``mask`` is true."""
    return compact(vertices, faces[mask])


def frustum(radius_bottom: float, radius_top: float, height: float, segments: int = 64) -> tuple:
    """Closed truncated cone along +Z, base at z=0 (vertices, faces). Used for cup collision hulls."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    bottom = np.hstack([ring * radius_bottom, np.zeros((segments, 1))])
    top = np.hstack([ring * radius_top, np.full((segments, 1), height)])
    vertices = np.vstack([bottom, top, [[0, 0, 0]], [[0, 0, height]]])
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [[i, j, segments + j], [i, segments + j, segments + i]]
        faces += [[2 * segments, j, i], [2 * segments + 1, segments + i, segments + j]]
    return vertices, np.array(faces)


def save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray | None = None) -> None:
    """Write a Wavefront OBJ, optionally with one texture coordinate per vertex.

    ``uvs`` follow this package's convention -- (0, 0) is the texture image's top-left corner, u runs
    along its columns and v down its rows -- and are flipped on write because OBJ measures v upward.
    """
    lines = [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in vertices]
    if uvs is not None:
        lines += [f"vt {u:.7f} {1.0 - v:.7f}" for u, v in uvs]
        lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in faces]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    Path(path).write_text("\n".join(lines) + "\n")


def grid_sheet(points: np.ndarray, uvs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate an (rows, cols, 3) grid of points into a sheet. Returns (vertices, faces, uvs).

    Faces wind counter-clockwise when the grid is seen with columns running right and rows running
    down, i.e. the sheet faces a camera that sees the grid the way an image is indexed.
    """
    rows, cols = points.shape[:2]
    index = np.arange(rows * cols).reshape(rows, cols)
    a, b, c, d = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()
    faces = np.concatenate([np.stack([a, d, c], axis=1), np.stack([a, c, b], axis=1)])
    return points.reshape(-1, 3), faces, uvs.reshape(-1, 2)
