"""Build the MuJoCo replica of the real teleop station.

The world frame is the follower arm's base frame: origin on the table under the base, +X the way
the arm reaches, +Z up, so the table top is the plane z = 0.

Everything measured from the real setup lives in ``calibration/results/`` (camera models, table and
room layout, object sizes, dynamics) and ``assets/textures/`` (surfaces photographed by the real
cameras and re-projected onto the model). ``build_spec`` only assembles them; to change what the
simulation looks like or how it moves, re-run the calibration scripts rather than editing numbers here.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.simulation import robot_visuals
from i2rt.simulation.cameras import PinholeCamera
from i2rt.simulation.mesh_utils import frustum, save_stl
from i2rt.simulation.table import TablePlane

SIM_ROOT = Path(__file__).resolve().parent
ASSETS = SIM_ROOT / "assets"
# Meshes derived from the robot models at build time (links cut where their paint changes, finger
# colliders, cups). They are regenerated whenever their source changes, so they live outside the package.
MESH_CACHE = Path(tempfile.gettempdir()) / "i2rt_simulation_meshes"
WRIST_MOUNT_MESH = SIM_ROOT.parent / "robot_models" / "station" / "assets" / "d405_wrist_linear_4310.stl"
RESULTS = SIM_ROOT / "calibration" / "results"

# Geom groups. MuJoCo draws groups 0-2 by default; the renderer switches the per-camera backdrops.
GROUP_VISUAL = robot_visuals.VISUAL_GROUP  # seen by every camera
GROUP_COLLISION = robot_visuals.COLLISION_GROUP  # never drawn
GROUP_OVERHEAD_ONLY = 4  # surfaces textured from the overhead camera's footage
GROUP_WRIST_ONLY = 5  # the same surfaces textured from the wrist camera's footage

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
BALL_HOLD = "ball_hold"
FINGER_JOINTS = ("joint7", "joint8")
FINGER_COLLIDERS = ("tip_left_finger", "tip_right_finger")  # the fingers alone (robot_visuals.add_finger_colliders)
FINGER_TRAVEL = 0.0475  # metres per finger, gripper command 1.0 = fully open


_JSON_CACHE: dict[Path, tuple[int, dict]] = {}


def load_json(name: str, default: dict | None = None) -> dict:
    """A calibration result from ``calibration/results/`` (re-read only when the file changes)."""
    path = RESULTS / name
    if not path.is_file():
        if default is None:
            raise FileNotFoundError(f"{path} is missing: run the calibration scripts (see README.md)")
        return default
    modified = path.stat().st_mtime_ns
    cached = _JSON_CACHE.get(path)
    if cached is None or cached[0] != modified:
        _JSON_CACHE[path] = cached = (modified, json.loads(path.read_text()))
    return cached[1]


@dataclass
class ObjectState:
    """Where a movable object starts: position of its base centre on the table, and heading."""

    xy: tuple[float, float]
    yaw: float = 0.0
    z: float = 0.0  # height of the object's lowest point above the table (the ball starts on a cup)


def object_pose(name: str, state: ObjectState, params: dict, table: TablePlane) -> tuple[np.ndarray, np.ndarray]:
    """World position and quaternion of an object's body frame for a state on the table.

    Cups have their frame at the centre of the rim they stand on; the ball's is its centre.
    """
    lift = state.z + (params["ball"]["radius"] if name == "ball" else 0.0) + 0.0005
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, table.quat, np.array(_yaw_quat(state.yaw)))
    return table.point(np.array(state.xy), lift), quat


@dataclass
class SceneOptions:
    """What to put on the table. Defaults reproduce a typical episode start."""

    objects: dict[str, ObjectState] = field(default_factory=dict)
    backdrop: bool = True
    """Photographed table/room surfaces. Off gives a plain grey table (faster, for dynamics work)."""
    include_objects: bool = True
    """Cups and ball. Off leaves the bare arm on the table (for fitting the arm's own dynamics)."""
    table: TablePlane | None = None
    """Table plane to build with. Defaults to the calibrated one (``calibration/results/table_plane.json``)."""

    def table_plane(self) -> TablePlane:
        return self.table if self.table is not None else TablePlane.load()

    wrist_camera_hardware: bool = True
    """Draw the wrist camera, its bracket and the cable on the arm."""
    timestep: float = 0.002


def _add_camera(body: mujoco.MjsBody, camera: PinholeCamera) -> None:
    cam = body.add_camera(name=camera.name)
    cam.pos = camera.pos
    cam.quat = camera.quat
    cam.resolution = [camera.width, camera.height]
    cam.sensor_size = [0.0064, 0.0048]
    cam.focal_pixel = [camera.fx, camera.fy]
    cam.principal_pixel = [camera.width / 2.0 - (camera.cx + 0.5), camera.height / 2.0 - (camera.cy + 0.5)]


# ---------------------------------------------------------------------------------------------------
# Measured objects
# ---------------------------------------------------------------------------------------------------

DEFAULT_OBJECTS = {
    # Two cups of the same make: fitted independently, the orange and the blue one agree to within a
    # few millimetres, so they share one size -- the mean of the two fits in ``objects.json``, which
    # replaces these starting values. ``base_recess`` is the depth of the hollow in a cup's base,
    # which is what a ball sits in when the cup stands upside down.
    "cup": {
        "rim_radius": 0.0375,
        "top_radius": 0.028,
        "height": 0.080,
        "mass": 0.020,
        "wall": 0.0025,
        "base_recess": 0.004,
    },
    # A soft ball-pit ball. Unoccluded overhead views put its radius at 27-29 mm; the gripper holds it
    # at finger gaps from 31 up to 56 mm, and for the loosest of those grasps to be a squeeze at all
    # it has to be 30 mm. The fingertips' soft contact lets the fingers sink in as the real ones do.
    "ball": {"radius": 0.030, "mass": 0.008},
    # The cardboard box at the right of the workspace (width, depth, height), fitted like the cups.
    "box": {"size": [0.123, 0.129, 0.075], "mass": 0.12},
    "colours": {
        "cup_orange": [1.0, 0.66, 0.16, 1.0],
        "cup_blue": [0.0, 0.50, 0.80, 1.0],
        "ball": [1.0, 0.50, 0.14, 1.0],
        "box": [0.80, 0.62, 0.47, 1.0],
    },
}
OBJECT_NAMES = ("cup_orange", "cup_blue", "ball", "box")


def object_parameters() -> dict:
    """Object sizes and colours: measured values where the calibration has produced them."""
    params = json.loads(json.dumps(DEFAULT_OBJECTS))
    measured = load_json("objects.json", default={})
    for key in ("cup", "ball", "box", "colours"):
        params[key].update({k: v for k, v in measured.get(key, {}).items() if k in params[key]})
    return params


def ball_rest_height(params: dict | None = None) -> float:
    """Height of the ball's lowest point when it sits on an upturned cup: the floor of the cup's base."""
    cup = (params or object_parameters())["cup"]
    return cup["height"] - cup["base_recess"]


def default_object_states() -> dict[str, ObjectState]:
    """A typical episode start: two cups in front of the arm, the ball on the orange one."""
    return {
        "cup_orange": ObjectState(xy=(0.42, 0.10)),
        "cup_blue": ObjectState(xy=(0.42, -0.12)),
        "ball": ObjectState(xy=(0.42, 0.10), z=ball_rest_height(DEFAULT_OBJECTS)),
        "box": ObjectState(xy=(0.351, -0.348), yaw=-0.07),
    }


def _cup_mesh(spec: mujoco.MjSpec, name: str, rim: float, top: float, height: float) -> None:
    if any(m.name == name for m in spec.meshes):
        return
    path = MESH_CACHE / f"{name}.stl"
    stamp = f"{rim:.5f} {top:.5f} {height:.5f}"
    marker = path.with_suffix(".stamp")
    if not path.is_file() or not marker.is_file() or marker.read_text() != stamp:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_stl(path, *frustum(rim, top, height, segments=72))
        marker.write_text(stamp)
    spec.add_mesh(name=name, file=str(path))


def _add_cup(
    spec: mujoco.MjSpec, name: str, state: ObjectState, params: dict, rgba: list[float], table: TablePlane
) -> None:
    """An upside-down cup: a free body whose frame sits at the centre of its rim, on the table.

    It is drawn at its full height. Its collider is the same cone cut ``base_recess`` lower, with a
    ring of small spheres standing on the edge up to the full height: together they are the hollow
    base of a real cup, which is what keeps a ball sitting on it instead of rolling off.
    """
    cup = params["cup"]
    recess = cup["base_recess"]
    floor_height = cup["height"] - recess
    floor_radius = cup["rim_radius"] + (cup["top_radius"] - cup["rim_radius"]) * floor_height / cup["height"]
    _cup_mesh(spec, "cup_visual", cup["rim_radius"], cup["top_radius"], cup["height"])
    _cup_mesh(spec, "cup_collision", cup["rim_radius"], floor_radius, floor_height)
    material = spec.add_material(name=f"{name}_plastic")
    material.rgba = rgba
    material.specular = 0.35
    material.shininess = 0.6
    pos, quat = object_pose(name, state, params, table)
    body = spec.worldbody.add_body(name=name, pos=pos, quat=quat)
    body.add_freejoint(name=f"{name}_free")
    shell = body.add_geom(
        name=f"{name}_shell",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cup_visual",
        material=f"{name}_plastic",
        group=GROUP_VISUAL,
        contype=0,
        conaffinity=0,
    )
    shell.mass = 0.0
    collider = body.add_geom(
        name=f"{name}_collision",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cup_collision",
        rgba=[*rgba[:3], 0.0],
        group=GROUP_COLLISION,
    )
    collider.mass = cup["mass"]
    collider.friction = [0.45, 0.003, 0.0002]
    collider.condim = 4
    collider.solref = [0.004, 1.0]
    ring_radius = cup["top_radius"] - cup["wall"]
    bead = recess / 2.0
    for i in range(20):
        angle = 2.0 * np.pi * i / 20
        rim = body.add_geom(
            name=f"{name}_rim{i}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[bead, 0, 0],
            pos=[ring_radius * np.cos(angle), ring_radius * np.sin(angle), floor_height + bead],
            rgba=[*rgba[:3], 0.0],
            group=GROUP_COLLISION,
        )
        rim.mass = 0.0
        rim.friction = [0.45, 0.003, 0.0002]
        rim.solref = [0.004, 1.0]


def _add_ball(spec: mujoco.MjSpec, state: ObjectState, params: dict, table: TablePlane) -> None:
    ball = params["ball"]
    material = spec.add_material(name="ball_plastic")
    material.rgba = params["colours"]["ball"]
    material.specular = 0.4
    material.shininess = 0.7
    pos, quat = object_pose("ball", state, params, table)
    body = spec.worldbody.add_body(name="ball", pos=pos, quat=quat)
    body.add_freejoint(name="ball_free")
    geom = body.add_geom(
        name="ball",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[ball["radius"], 0, 0],
        rgba=[1.0, 0.5, 0.1, 0.0],
        group=GROUP_COLLISION,
    )
    geom.mass = ball["mass"]
    geom.condim = 6  # torsional + rolling friction, so it settles instead of rolling forever
    geom.friction = [0.6, 0.003, 0.0006]
    geom.solref = [0.004, 1.0]
    # What is drawn is a separate ellipsoid, a sphere until the gripper squeezes the ball: the
    # environment then flattens it between the fingers, as the real soft ball is (env.py).
    visual = body.add_geom(
        name="ball_visual",
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        size=[ball["radius"]] * 3,
        material="ball_plastic",
        group=GROUP_VISUAL,
        contype=0,
        conaffinity=0,
    )
    visual.mass = 0.0


def _add_box(spec: mujoco.MjSpec, state: ObjectState, params: dict, table: TablePlane) -> None:
    """The cardboard box: a free body whose frame is the centre of its base."""
    size = np.asarray(params["box"]["size"])
    material = spec.add_material(name="cardboard")
    material.rgba = params["colours"]["box"]
    material.specular = 0.02
    material.shininess = 0.05
    pos, quat = object_pose("box", state, params, table)
    body = spec.worldbody.add_body(name="box", pos=pos, quat=quat)
    body.add_freejoint(name="box_free")
    geom = body.add_geom(
        name="box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size / 2.0,
        pos=[0, 0, size[2] / 2.0],
        material="cardboard",
        group=GROUP_VISUAL,
    )
    geom.mass = params["box"]["mass"]
    geom.friction = [0.5, 0.005, 0.0005]
    geom.condim = 4
    geom.solref = [0.004, 1.0]


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]


# ---------------------------------------------------------------------------------------------------
# Static surroundings
# ---------------------------------------------------------------------------------------------------


_UNLIT_RGBA = 1e-3


def _add_textured_sheet(spec: mujoco.MjSpec, name: str, mesh: Path, texture: Path, group: int, lit: bool) -> None:
    """A photographed surface: a sheet mesh carrying a texture re-projected from real footage."""
    spec.add_texture(name=f"{name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(texture))
    material = spec.add_material(name=f"{name}_mat")
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_tex"
    material.specular = 0.0
    material.shininess = 0.0
    # A lit sheet receives the scene's lights, which are balanced to sum to one on a horizontal
    # surface, so it shows its photograph plus simulated shadows. An unlit sheet must show its
    # photograph exactly whatever the lights do: MuJoCo's emission is ``emission * rgba`` while
    # ambient and diffuse are ``rgba``, so a near-zero rgba with a matching large emission leaves
    # the emitted texture at full strength and the lit terms at nothing.
    if lit:
        material.emission = 0.0
        material.rgba = [1.0, 1.0, 1.0, 1.0]
    else:
        material.emission = 1.0 / _UNLIT_RGBA
        material.rgba = [_UNLIT_RGBA, _UNLIT_RGBA, _UNLIT_RGBA, 1.0]
    sheet = spec.add_mesh(name=f"{name}_mesh", file=str(mesh))
    sheet.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{name}_mesh",
        material=f"{name}_mat",
        contype=0,
        conaffinity=0,
        group=group,
    )


def _add_surroundings(spec: mujoco.MjSpec, options: SceneOptions) -> None:
    layout = load_json("overhead_layout.json", default={})
    far_x = layout.get("table_far_edge_x", 0.78)
    right_y = layout.get("table_right_edge_y", -0.49)
    # The table the arm is bolted to, as the arm sees it (table.py). Only its top is ever touched;
    # the slab extends past what the cameras see.
    plane = options.table_plane()
    half = np.array([(far_x + 0.12) / 2.0, (1.45 - right_y) / 2.0, 0.015])
    centre_xy = np.array([(far_x - 0.12) / 2.0, (1.45 + right_y) / 2.0])
    table = spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=plane.point(centre_xy, -half[2]),
        quat=plane.quat,
        rgba=[0.42, 0.43, 0.45, 1.0],
        group=GROUP_COLLISION if (options.backdrop and layout) else GROUP_VISUAL,
    )
    table.friction = [0.45, 0.003, 0.0002]
    table.solref = [0.004, 1.0]
    if options.backdrop and layout:
        _add_textured_sheet(
            spec,
            "overhead_table",
            ASSETS / layout["table_mesh"],
            ASSETS / layout["table_texture"]["file"],
            GROUP_OVERHEAD_ONLY,
            lit=True,
        )
        _add_textured_sheet(
            spec,
            "overhead_backdrop",
            ASSETS / layout["backdrop_mesh"],
            ASSETS / layout["backdrop_texture"],
            GROUP_OVERHEAD_ONLY,
            lit=False,
        )
    wrist_layout = load_json("wrist_layout.json", default={})
    if options.backdrop:
        for sheet in wrist_layout.get("sheets", []):
            _add_textured_sheet(
                spec,
                sheet["name"],
                ASSETS / sheet["mesh"],
                ASSETS / sheet["texture"],
                GROUP_WRIST_ONLY,
                lit=sheet.get("lit", False),
            )


DEFAULT_LIGHTING = {
    "ambient": 0.62,  # share of a horizontal surface's light that comes from everywhere
    "count": 5,  # directional lights in the cluster
    "spread_deg": 16.0,  # how far the cluster's lights fan out around its main direction
    "lean_deg": 0.0,  # how far the main direction leans from straight down...
    "heading_deg": 0.0,  # ...and towards which way shadows then fall (0 = +X, 90 = +Y)
    "specular": 0.25,
    "shadow_strength": 0.85,
    "shadow_blur": 7.0,
}


def lighting_parameters() -> dict:
    return {**DEFAULT_LIGHTING, **load_json("lighting.json", default={})}


def light_directions(lighting: dict) -> np.ndarray:
    """Unit travel directions of the light cluster (count, 3): one along the main direction, the rest fanned around it."""
    lean, heading, spread = (np.radians(lighting[k]) for k in ("lean_deg", "heading_deg", "spread_deg"))
    main = np.array([np.sin(lean) * np.cos(heading), np.sin(lean) * np.sin(heading), -np.cos(lean)])
    side = np.cross(main, [0.0, 1.0, 0.0] if abs(main[1]) < 0.9 else [1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    other = np.cross(main, side)
    count = int(lighting["count"])
    directions = [main]
    for i in range(1, count):
        around = 2.0 * np.pi * i / max(count - 1, 1)
        fanned = np.cos(spread) * main + np.sin(spread) * (np.cos(around) * side + np.sin(around) * other)
        directions.append(fanned / np.linalg.norm(fanned))
    return np.array(directions)


def light_strengths(lighting: dict, directions: np.ndarray) -> np.ndarray:
    """Diffuse intensity of each light such that ambient plus all of them is exactly 1 on a horizontal surface."""
    share = (1.0 - lighting["ambient"]) / len(directions)
    return share / np.maximum(-directions[:, 2], 0.2)


def _add_lights(spec: mujoco.MjSpec) -> None:
    """Soft office lighting: mostly ambient, plus a cluster of ceiling lights whose shadows blend into a penumbra.

    The components are balanced so a horizontal, upward-facing surface receives exactly 1.0: lit
    photographed surfaces (the table) then reproduce their photograph wherever nothing shades them.
    Where the cluster points -- which way shadows fall -- is fitted to the footage (``calibration/lighting.py``).
    """
    lighting = lighting_parameters()
    spec.visual.headlight.ambient = [lighting["ambient"]] * 3
    spec.visual.headlight.diffuse = [0.0] * 3
    spec.visual.headlight.specular = [0.0] * 3
    directions = light_directions(lighting)
    for i, (direction, strength) in enumerate(zip(directions, light_strengths(lighting, directions), strict=True)):
        light = spec.worldbody.add_light(name=f"ceiling{i}", pos=[0.3, 0.0, 2.5], dir=direction.tolist())
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.diffuse = [float(strength)] * 3
        light.specular = [lighting["specular"] / len(directions)] * 3
        light.castshadow = True


def apply_lighting(model: mujoco.MjModel, lighting: dict) -> None:
    """Re-aim and re-balance the light cluster of a compiled model (the count must not change)."""
    directions = light_directions(lighting)
    strengths = light_strengths(lighting, directions)
    model.vis.headlight.ambient[:] = lighting["ambient"]
    for i, (direction, strength) in enumerate(zip(directions, strengths, strict=True)):
        light = model.light(f"ceiling{i}").id
        model.light_dir[light] = direction
        model.light_diffuse[light] = strength
        model.light_specular[light] = lighting["specular"] / len(directions)


# ---------------------------------------------------------------------------------------------------
# Robot dynamics
# ---------------------------------------------------------------------------------------------------

DEFAULT_DYNAMICS = {
    # Motor-side PD gains the real arm runs (i2rt/robots/config/yam_v1.yml) and joint-space
    # mechanical parameters. The mechanical ones are fitted to recorded command/response pairs by
    # calibration/dynamics.py; these are the starting values.
    "kp": [80.0, 80.0, 80.0, 10.0, 10.0, 10.0],
    "kd": [5.0, 5.0, 5.0, 1.5, 1.5, 1.5],
    "gravity_comp_factor": [1.0, 1.1, 1.1, 1.2, 1.0, 1.0],
    "armature": [0.05, 0.05, 0.05, 0.005, 0.005, 0.005],
    "damping": [0.5, 0.5, 0.5, 0.05, 0.05, 0.05],
    "frictionloss": [0.3, 0.3, 0.3, 0.06, 0.06, 0.06],
    "torque_limit": [28.0, 28.0, 28.0, 10.0, 10.0, 10.0],
    "finger_kp": 4000.0,
    "finger_kv": 60.0,
    "finger_force": 25.0,
    "finger_armature": 0.02,
    "finger_damping": 8.0,
    "finger_contact_timeconst": 0.03,
    "command_delay_s": 0.0,
}


def dynamics_parameters() -> dict:
    params = dict(DEFAULT_DYNAMICS)
    params.update(load_json("dynamics.json", default={}))
    return params


def _add_robot_dynamics(spec: mujoco.MjSpec) -> None:
    """Joint mechanics and the motors' own PD loops.

    The real arm sends each motor a position target, kp, kd and a feed-forward torque; the motor
    closes the PD loop itself at kHz rates. A MuJoCo position actuator with ``kv`` is that loop
    (integrated implicitly, so it is stable at the simulation step). The gravity feed-forward is
    added by the controller in ``env.py``, as on the real arm.
    """
    dyn = dynamics_parameters()
    for i, name in enumerate(ARM_JOINTS):
        joint = spec.joint(name)
        joint.armature = dyn["armature"][i]
        joint.damping = [dyn["damping"][i], 0.0, 0.0]  # linear term of MuJoCo's damping polynomial
        joint.frictionloss = dyn["frictionloss"][i]
        actuator = spec.add_actuator(name=f"motor{i + 1}", target=name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.set_to_position(kp=dyn["kp"][i], kv=dyn["kd"][i])
        actuator.forcelimited = True
        actuator.forcerange = [-dyn["torque_limit"][i], dyn["torque_limit"][i]]
    for name in FINGER_JOINTS:
        joint = spec.joint(name)
        joint.armature = dyn["finger_armature"]
        joint.damping = [dyn["finger_damping"], 0.0, 0.0]
        actuator = spec.add_actuator(name=f"finger_{name}", target=name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.set_to_position(kp=dyn["finger_kp"], kv=dyn["finger_kv"])
        actuator.forcelimited = True
        actuator.forcerange = [-dyn["finger_force"], dyn["finger_force"]]
    for geom in spec.geoms:
        if geom.name in FINGER_COLLIDERS:
            geom.friction = [1.2, 0.02, 0.001]
            geom.condim = 4
            geom.solref = [
                0.004,
                1.0,
            ]  # rigid against the table, the cups and the box; the ball is the exception below


# ---------------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------------


def load_cameras() -> dict[str, PinholeCamera]:
    """The calibrated cameras, keyed by the names the dataset uses."""
    cameras = {"overhead": PinholeCamera.load(RESULTS / "overhead_camera.json")}
    wrist = RESULTS / "wrist_camera.json"
    if wrist.is_file():
        cameras["right_wrist"] = PinholeCamera.load(wrist)
    return cameras


def build_spec(options: SceneOptions | None = None) -> mujoco.MjSpec:
    options = options or SceneOptions()
    source_xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    spec = mujoco.MjSpec.from_file(source_xml)
    spec.modelname = "yam_station"
    spec.option.timestep = options.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0  # stiffer friction, so grasped objects do not creep out of the fingers
    spec.option.gravity = (-9.81 * options.table_plane().normal).tolist()  # straight down onto the table (table.py)
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1440
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 4
    spec.visual.map.znear = 0.002  # the wrist camera sits a few centimetres from the fingers
    spec.visual.map.zfar = 12.0
    spec.stat.extent = 1.0
    spec.stat.center = [0.3, 0.0, 0.2]

    robot_visuals.add_materials(spec)
    robot_visuals.paint_robot(spec, MESH_CACHE)
    robot_visuals.add_finger_colliders(spec, Path(source_xml), MESH_CACHE)
    if options.wrist_camera_hardware:
        robot_visuals.add_wrist_camera_hardware(spec, WRIST_MOUNT_MESH, group=GROUP_OVERHEAD_ONLY)
        robot_visuals.add_cable(spec)
    _add_robot_dynamics(spec)
    _add_surroundings(spec, options)
    _add_lights(spec)

    cameras = load_cameras()
    _add_camera(spec.worldbody, cameras["overhead"])
    if "right_wrist" in cameras:
        _add_camera(spec.body(cameras["right_wrist"].parent_body), cameras["right_wrist"])

    if options.include_objects:
        params = object_parameters()
        states = {**default_object_states(), **options.objects}
        for name in ("cup_orange", "cup_blue"):
            _add_cup(spec, name, states[name], params, params["colours"][name], options.table_plane())
        _add_ball(spec, states["ball"], params, options.table_plane())
        _add_box(spec, states["box"], params, options.table_plane())
        # Dormant tie between the gripper and the ball, switched on by the environment while both
        # fingers squeeze the ball (see ``YamStationEnv._update_ball_hold``).
        hold = spec.add_equality(
            name=BALL_HOLD,
            type=mujoco.mjtEq.mjEQ_CONNECT,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1="gripper",
            name2="ball",
        )
        hold.active = False
        hold.solref = [0.01, 1.0]
        # The ball gives where the fingers press it: it is a soft shell the real gripper squeezes to
        # as little as half its width. A long contact time constant lets the fingers sink in instead
        # of stalling on a rigid sphere. Only this pair is soft -- the same fingers stop dead on the
        # table -- so it is declared explicitly rather than left to the geoms' own properties.
        timeconst = dynamics_parameters()["finger_contact_timeconst"]
        for finger in FINGER_COLLIDERS:
            pair = spec.add_pair(name=f"{finger}_on_ball", geomname1=finger, geomname2="ball")
            pair.condim = 4
            pair.friction = [1.2, 1.2, 0.02, 0.001, 0.001]
            pair.solref = [timeconst, 1.0]
    for site in spec.sites:
        site.rgba = [0.0, 0.0, 0.0, 0.0]  # tool-centre markers from the gripper model
    return spec


def build_gravity_model() -> mujoco.MjModel:
    """The plain arm + gripper model the real controller computes its gravity feed-forward from."""
    return mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))


def build_model(options: SceneOptions | None = None) -> mujoco.MjModel:
    return build_spec(options).compile()
