"""Make the simulated arm look like the real one: two-tone links, wrist camera, cable, LEDs.

The shipped arm model has one debug-coloured STL per link. The real YAM is black (base, motor
housings, elbow fork, wrist, gripper, fingers) with white shells on the upper arm and the forearm,
so each two-tone link is cut at the planes where its shell meets a motor housing. The planes were
read off the meshes themselves: every housing ends in a large flat face normal to the link axis.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from i2rt.simulation.mesh_utils import load_stl, save_stl, split_by_plane

# Per link: cuts along the mesh's own Z axis (metres) and the paint of each resulting slab, low to high.
#   link2 (upper arm): shoulder motor housing | white shell | elbow motor housing
#   link3 (forearm):   wrist motor housing    | white shell | elbow fork
LINK_PAINT: dict[str, tuple[tuple[float, ...], tuple[str, ...]]] = {
    "base": ((), ("black",)),
    "link1": ((), ("black",)),
    "link2": ((0.0529, 0.2461), ("black", "white", "black")),
    "link3": ((0.0719, 0.2500), ("black", "white", "black")),
    "link4": ((), ("black",)),
    "link5": ((), ("black",)),
    "gripper": ((), ("black",)),
    "tip_left": ((), ("finger",)),
    "tip_right": ((), ("finger",)),
}

VISUAL_GROUP = 2  # drawn by every camera
COLLISION_GROUP = 3  # never drawn
ROBOT_CONTYPE = 2


def add_materials(spec: mujoco.MjSpec) -> None:
    """Paints for the arm. Values are tuned against the overhead footage (see calibration/appearance.py)."""
    for name, rgba, specular, shininess in (
        # The black housings are satin: faces turned up to the ceiling lights read distinctly lighter
        # than their sides. A strong, very broad specular lobe gives that; a tight one would not, the
        # camera being 40 degrees off the mirror direction.
        ("robot_black", (0.085, 0.085, 0.09, 1.0), 0.9, 0.06),
        ("robot_white", (0.86, 0.86, 0.85, 1.0), 0.25, 0.5),
        ("robot_finger", (0.07, 0.07, 0.075, 1.0), 0.5, 0.08),
        ("camera_bracket", (0.33, 0.50, 0.72, 1.0), 0.1, 0.1),
        ("camera_body", (0.06, 0.06, 0.065, 1.0), 0.4, 0.5),
        ("cable", (0.05, 0.05, 0.05, 1.0), 0.1, 0.1),
    ):
        material = spec.add_material(name=name)
        material.rgba = rgba
        material.specular = specular
        material.shininess = shininess


def paint_robot(spec: mujoco.MjSpec, mesh_cache: Path) -> None:
    """Replace each link's debug-coloured geom with painted visual pieces plus a hidden collider.

    The original geom keeps doing collision (now invisible, in ``COLLISION_GROUP``); the painted
    pieces are visual only. Cut meshes are cached as STL files under ``mesh_cache`` so the scene can
    be exported as a standalone MJCF.
    """
    mesh_cache.mkdir(parents=True, exist_ok=True)
    mesh_files = {m.name: Path(m.file) for m in spec.meshes}
    for geom in list(spec.geoms):
        link = geom.meshname
        if link not in LINK_PAINT:
            continue
        cuts, paints = LINK_PAINT[link]
        body = geom.parent
        pieces = _cut_link(mesh_files[link], cuts, mesh_cache, link)
        for index, (path, paint) in enumerate(zip(pieces, paints, strict=True)):
            mesh_name = f"{link}_paint{index}"
            mesh = spec.add_mesh(name=mesh_name, file=str(path))
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL  # open surfaces after a cut
            visual = body.add_geom(
                name=f"{link}_visual{index}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                pos=geom.pos,
                quat=geom.quat,
                material=f"robot_{paint}",
                contype=0,
                conaffinity=0,
                group=VISUAL_GROUP,
                mass=0.0,
            )
            # Each body carries an explicit <inertial>, so visual geoms must not add mass of their own.
            visual.density = 0.0
        geom.name = f"{link}_collision"
        geom.group = COLLISION_GROUP
        geom.rgba = [0.5, 0.5, 0.5, 0.0]
        # The colliders are convex hulls of whole links, which overlap their neighbours at every
        # joint. Type 2 / affinity 1 makes the arm collide with the world (type 1) but not itself.
        if geom.contype or geom.conaffinity:
            geom.contype = ROBOT_CONTYPE
            geom.conaffinity = 1


def _cut_link(source: Path, cuts: tuple[float, ...], cache: Path, link: str) -> list[Path]:
    """Slabs of ``source`` between consecutive Z cuts, written to (and reused from) ``cache``."""
    paths = [cache / f"{link}_part{i}.stl" for i in range(len(cuts) + 1)]
    stamp = cache / f"{link}.stamp"
    key = f"{source.stat().st_mtime_ns}:{source.stat().st_size}:{cuts}"
    if all(p.is_file() for p in paths) and stamp.is_file() and stamp.read_text() == key:
        return paths
    vertices, faces = load_stl(source)
    remaining = (vertices, faces)
    for path, z in zip(paths, cuts, strict=False):
        remaining, below = split_by_plane(*remaining, np.array([0.0, 0.0, z]), np.array([0.0, 0.0, 1.0]))
        save_stl(path, *below)
    save_stl(paths[-1], *remaining)
    stamp.write_text(key)
    return paths


FINGER_ROOT_Z = 0.074  # metres along the gripper's forward axis where a finger leaves its rack


def add_finger_colliders(spec: mujoco.MjSpec, source_xml: Path, mesh_cache: Path) -> None:
    """Give each finger a collider of the finger alone.

    A ``tip_*`` mesh is the finger *and* the toothed rack that drives it, and the rack reaches across
    the gripper's centre line. MuJoCo collides meshes by their convex hull, and the hull of finger
    plus rack is a wedge that fills the very gap an object is grasped in. So the rack is cut off (it
    never touches anything: it lives inside the gripper body) and only the finger's hull collides.
    """
    plain = mujoco.MjModel.from_xml_path(str(source_xml))
    data = mujoco.MjData(plain)
    mujoco.mj_forward(plain, data)
    mount_pos = data.body("gripper").xpos
    mount_rot = data.body("gripper").xmat.reshape(3, 3)
    mesh_cache.mkdir(parents=True, exist_ok=True)
    for tip in ("tip_left", "tip_right"):
        geom_id = next(g for g in range(plain.ngeom) if plain.body(plain.geom_bodyid[g]).name == tip)
        mesh_id = plain.geom_dataid[geom_id]
        vertices = plain.mesh_vert[
            plain.mesh_vertadr[mesh_id] : plain.mesh_vertadr[mesh_id] + plain.mesh_vertnum[mesh_id]
        ].astype(float)
        faces = plain.mesh_face[
            plain.mesh_faceadr[mesh_id] : plain.mesh_faceadr[mesh_id] + plain.mesh_facenum[mesh_id]
        ]
        world = data.geom_xpos[geom_id] + vertices @ data.geom_xmat[geom_id].reshape(3, 3).T
        # Cut in the gripper's frame, where "forward" is +Z whatever the finger's own frame is.
        in_mount = (world - mount_pos) @ mount_rot
        (finger, finger_faces), _ = split_by_plane(
            in_mount, faces, np.array([0.0, 0.0, FINGER_ROOT_Z]), np.array([0.0, 0.0, 1.0])
        )
        body_pos, body_rot = data.body(tip).xpos, data.body(tip).xmat.reshape(3, 3)
        in_body = (mount_pos + finger @ mount_rot.T - body_pos) @ body_rot
        path = mesh_cache / f"{tip}_finger.stl"
        save_stl(path, in_body, finger_faces)
        mesh = spec.add_mesh(name=f"{tip}_finger", file=str(path))
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        whole = next(g for g in spec.geoms if g.name == f"{tip}_collision")
        whole.contype, whole.conaffinity = 0, 0
        collider = spec.body(tip).add_geom(
            name=f"{tip}_finger",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{tip}_finger",
            rgba=[0.5, 0.5, 0.5, 0.0],
            contype=ROBOT_CONTYPE,
            conaffinity=1,
            group=COLLISION_GROUP,
            mass=0.0,
        )
        collider.density = 0.0


# The printed wrist-camera mount, as modelled for I2RT's own camera station (optical frame of the
# camera it was designed for, in the gripper mount frame). The real setup carries the same style of
# mount, printed in light blue, with a different camera in it; the fitted wrist camera sits within a
# centimetre of this frame.
WRIST_MOUNT_POS = (0.0, -0.0704, 0.0770)
WRIST_MOUNT_QUAT = (-0.9762948880484529, 0.2164446616738001, 0.0, 0.0)
WRIST_HARDWARE_MASS = 0.10  # camera, mount and the cable's share: carried by the arm, unknown to its controller


def add_wrist_camera_hardware(spec: mujoco.MjSpec, mount_mesh: Path, group: int = VISUAL_GROUP) -> None:
    """Draw the wrist camera's printed mount on the gripper, and give the gripper its weight.

    ``group`` is the geom group the mount is drawn in. The wrist camera sits inside the mount, so the
    mount must be in a group that camera does not draw, or it would fill the camera's own view.
    """
    mesh = spec.add_mesh(name="wrist_camera_mount", file=str(mount_mesh))
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    body = spec.body("gripper").add_body(name="wrist_camera_hardware", pos=WRIST_MOUNT_POS, quat=WRIST_MOUNT_QUAT)
    body.mass = WRIST_HARDWARE_MASS
    body.ipos = [0.0, 0.0, -0.02]
    body.inertia = [4e-5, 4e-5, 4e-5]
    body.explicitinertial = True
    visual = body.add_geom(
        name="wrist_camera_mount",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="wrist_camera_mount",
        material="camera_bracket",
        contype=0,
        conaffinity=0,
        group=group,
        mass=0.0,
    )
    visual.density = 0.0


# The wrist camera's USB cable is tied along the arm. Per link: (start, end) of the run in that
# link's *mesh* coordinates, on the face that looks back at the overhead camera, standing just
# proud of the shell. Purely visual: a black line down each white link is a strong cue in the
# overhead view.
CABLE_RUNS = {
    "link3": ((-0.0165, 0.047, 0.085), (-0.0165, 0.047, 0.238)),
    "link2": ((0.0462, 0.048, 0.062), (0.0462, 0.048, 0.238)),
}
CABLE_RADIUS = 0.0019


def add_cable(spec: mujoco.MjSpec) -> None:
    """Draw the camera cable as a capsule along the forearm and the upper arm."""
    for link, (start, end) in CABLE_RUNS.items():
        shell = next(g for g in spec.geoms if g.name == f"{link}_collision")
        offset = np.asarray(shell.pos)  # the link meshes are placed by translation only
        cable = spec.body(link).add_geom(
            name=f"{link}_cable",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*(offset + np.asarray(start)), *(offset + np.asarray(end))],
            size=[CABLE_RADIUS, 0, 0],
            material="cable",
            contype=0,
            conaffinity=0,
            group=VISUAL_GROUP,
            mass=0.0,
        )
        cable.density = 0.0
