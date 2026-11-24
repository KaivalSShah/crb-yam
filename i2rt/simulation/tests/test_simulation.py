"""Tests for the simulated station. Physics and geometry run anywhere; rendering needs an OpenGL context."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

# The simulation needs OpenCV, which the package's core dependencies do not include (the teleop
# recorder in i2rt.robot_learning needs it too). Without it there is nothing to test here.
pytest.importorskip("cv2", reason="i2rt.simulation needs opencv-python (uv pip install opencv-python-headless)")

from i2rt.simulation import mesh_utils, scene
from i2rt.simulation.cameras import PinholeCamera, look_at_quat
from i2rt.simulation.env import COMMAND_LIMIT_BUFFER, IMAGE_KEY, STATE_KEY, EnvConfig, YamStationEnv
from i2rt.simulation.table import TablePlane


@pytest.fixture(scope="module")
def env() -> YamStationEnv:
    return YamStationEnv(EnvConfig(cameras=()))


def _renderer(model: mujoco.MjModel, height: int, width: int) -> mujoco.Renderer:
    try:
        return mujoco.Renderer(model, height, width)
    except Exception as error:  # no display / GL backend on this machine
        pytest.skip(f"no OpenGL context for offscreen rendering: {error}")


# ---- geometry ------------------------------------------------------------------------------------------


def test_plane_cut_keeps_every_bit_of_surface() -> None:
    vertices, faces = mesh_utils.frustum(0.04, 0.03, 0.08, segments=24)

    def area(v: np.ndarray, f: np.ndarray) -> float:
        t = v[f]
        return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())

    (upper, upper_faces), (lower, lower_faces) = mesh_utils.split_by_plane(
        vertices, faces, np.array([0, 0, 0.031]), np.array([0, 0, 1.0])
    )
    assert area(upper, upper_faces) + area(lower, lower_faces) == pytest.approx(area(vertices, faces), rel=1e-9)
    assert upper[:, 2].min() >= 0.031 - 1e-9 and lower[:, 2].max() <= 0.031 + 1e-9


def test_table_plane_is_self_consistent() -> None:
    table = TablePlane(0.02, 0.05, -0.01)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, table.quat)
    assert np.allclose(rotation.reshape(3, 3)[:, 2], table.normal)
    point = table.point(np.array([0.4, 0.1]), lift=0.05)
    assert table.lift_of(point) == pytest.approx(0.05)
    hit = table.intersect(np.array([-0.3, 0.0, 0.8]), np.array([[0.5, 0.1, -0.7]]))[0]
    assert hit[2] == pytest.approx(table.height(hit[0], hit[1]))


def test_camera_ray_round_trip() -> None:
    position = np.array([-0.3, 0.05, 0.8])
    camera = PinholeCamera(
        "c",
        position.tolist(),
        look_at_quat(position, np.array([0.3, 0.0, 0.0])).tolist(),
        fx=380.0,
        fy=508.0,
        cx=321.0,
        cy=238.0,
        k1=-0.08,
    )
    points = np.array([[0.3, 0.0, 0.0], [0.6, -0.2, 0.0], [0.1, 0.25, 0.0]])
    pixels = camera.project(points)[:, :2]
    assert np.allclose(camera.pixel_to_plane(pixels), points, atol=1e-6)


@pytest.mark.parametrize("k1", [0.0, -0.2])
def test_rendered_points_land_where_the_camera_model_says(k1: float) -> None:
    points = np.array([[0.3, 0.0, 0.0], [0.5, 0.25, 0.05], [0.1, -0.3, 0.2], [0.0, 0.3, 0.1]])
    spheres = "".join(
        f'<geom name="s{i}" type="sphere" size="0.006" pos="{p[0]} {p[1]} {p[2]}"/>' for i, p in enumerate(points)
    )
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><visual><global offwidth="2000" offheight="1500"/></visual><worldbody>{spheres}'
        '<camera name="c" pos="0 0 1" resolution="640 480" focalpixel="500 500" sensorsize="0.0064 0.0048"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    position = np.array([-0.3, 0.05, 0.75])
    camera = PinholeCamera(
        "c",
        position.tolist(),
        look_at_quat(position, np.array([0.3, 0.0, 0.0]), roll=0.03).tolist(),
        fx=320.0,
        fy=427.0,
        cx=323.0,
        cy=236.0,
        k1=k1,
    )
    canvas, size = 1.4, (1344, 1792)
    camera.apply_to_model(model, 0, canvas_scale=canvas)
    mujoco.mj_forward(model, data)
    renderer = _renderer(model, *size)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="c")
    segmentation = renderer.render()[..., 0].astype(np.float32)
    import cv2

    map_x, map_y = camera.distortion_maps(size[1], size[0], canvas_scale=canvas)
    seen = cv2.remap(segmentation, map_x, map_y, cv2.INTER_NEAREST, borderValue=-1)
    expected = camera.project(points)
    for i in range(len(points)):
        rows, cols = np.nonzero(seen == model.geom(f"s{i}").id)
        assert len(cols) > 0
        assert np.allclose([cols.mean(), rows.mean()], expected[i, :2], atol=0.6)


# ---- scene and physics ---------------------------------------------------------------------------------


def test_fingers_open_to_the_gripper_stroke_and_close_to_touching(env: YamStationEnv) -> None:
    model, data = env.model, env.data
    left, right = model.geom("tip_left_finger").id, model.geom("tip_right_finger").id
    gaps = []
    for opening in (0.0, 1.0):
        env.reset(np.array([0, 0, 0, 0, 0, 0, opening]))
        gaps.append(mujoco.mj_geomDistance(model, data, left, right, 0.3, np.zeros(6)))
    assert abs(gaps[0]) < 0.003  # closed: pads meet
    assert gaps[1] == pytest.approx(2 * scene.FINGER_TRAVEL, abs=0.004)  # open: the full stroke


def test_objects_come_to_rest_where_they_are_put(env: YamStationEnv) -> None:
    env.reset()
    for _ in range(60):
        env.advance(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    params = scene.object_parameters()
    states = scene.default_object_states()
    for cup in ("cup_orange", "cup_blue"):
        position = env.object_position(cup)
        assert abs(env.table.lift_of(position)) < 0.003
        assert np.linalg.norm(position[:2] - states[cup].xy) < 0.005
    ball = env.object_position("ball")
    assert env.table.lift_of(ball) == pytest.approx(
        scene.ball_rest_height(params) + params["ball"]["radius"], abs=0.004
    )
    assert np.linalg.norm(ball[:2] - states["ball"].xy) < params["cup"]["top_radius"]


def test_fingertips_stop_on_the_table(env: YamStationEnv) -> None:
    """The fingers give against the ball only: pushed onto the table they stop at its surface."""
    model, data = env.model, env.data
    fingers = [model.geom(name).id for name in scene.FINGER_COLLIDERS]
    away = {
        "cup_orange": scene.ObjectState(xy=(0.35, 0.80)),
        "ball": scene.ObjectState(xy=(0.35, 0.80)),
        "cup_blue": scene.ObjectState(xy=(0.50, 0.80)),
        "box": scene.ObjectState(xy=(0.45, 1.00)),
    }

    def clearance() -> float:
        """Height of the fingers' lowest point over the table."""
        heights = []
        for finger in fingers:
            mesh = model.geom_dataid[finger]
            vertices = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            world = data.geom_xpos[finger] + vertices @ data.geom_xmat[finger].reshape(3, 3).T
            heights.append(min(env.table.lift_of(point) for point in world))
        return min(heights)

    # A recorded grasp pose, gripper closed, with the shoulder lowered until the fingertips are just over the table.
    pose = np.array([-0.5, 1.17, 0.76, -0.40, 0.10, -0.05, 0.0])
    env.reset(pose, objects=away)
    while clearance() > 0.015:
        pose[1] += 0.01
        env.reset(pose, objects=away)
    target = pose.copy()
    target[1] += 0.30  # ask for fingertips some 8 cm below the table top
    for _ in range(60):
        env.advance(target)
    assert -0.002 < clearance() < 0.001  # they are on it, not in it
    assert target[1] - env.joint_state()[1] > 0.1  # and the arm is held up short of its target


def test_arm_follows_position_commands(env: YamStationEnv) -> None:
    env.reset()
    target = np.array([0.3, 1.2, 0.9, -0.2, 0.1, -0.1, 0.5])
    for _ in range(120):  # the real teleop ramps targets; a 4 s hold is plenty to arrive
        env.advance(target)
    state = env.joint_state()
    assert np.abs(state[:6] - target[:6]).max() < 0.06
    assert state[6] == pytest.approx(target[6], abs=0.03)


def test_arm_commands_are_clipped_as_the_real_controller_clips_them(env: YamStationEnv) -> None:
    """``get_yam_robot`` lets a position command reach 0.15 rad past a joint limit and no further."""
    env.reset()
    upper = np.array([env.model.joint(name).range[1] for name in scene.ARM_JOINTS])
    env.advance(np.append(upper + 1.0, 0.5))
    sent = np.array([env.data.actuator(f"motor{i + 1}").ctrl[0] for i in range(len(scene.ARM_JOINTS))])
    assert np.allclose(sent, upper + COMMAND_LIMIT_BUFFER)


def test_random_commands_leave_the_simulation_sane(env: YamStationEnv) -> None:
    """A policy can ask for anything. Ten seconds of uniformly random targets: nothing diverges."""
    model, data = env.model, env.data
    limits = np.array([model.joint(name).range for name in scene.ARM_JOINTS])
    rng = np.random.default_rng(0)
    env.reset()
    for _ in range(300):
        env.advance(np.append(rng.uniform(limits[:, 0], limits[:, 1]), rng.uniform(0.0, 1.0)))
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    joints = env.joint_state()[:6]
    assert (joints > limits[:, 0] - 0.05).all() and (joints < limits[:, 1] + 0.05).all()
    assert sum(warning.number for warning in data.warning) == 0  # MuJoCo raised no instability warning


def test_env_speaks_the_dataset_format() -> None:
    cameras = tuple(scene.load_cameras())
    env = YamStationEnv(EnvConfig(cameras=cameras))
    try:
        observation = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    except Exception as error:
        if "gl" in str(error).lower() or "display" in str(error).lower():
            pytest.skip(f"no OpenGL context for offscreen rendering: {error}")
        raise
    assert observation[STATE_KEY].shape == (7,) and observation[STATE_KEY].dtype == np.float32
    for camera in cameras:
        image = observation[IMAGE_KEY.format(camera)]
        assert image.shape == (480, 640, 3) and image.dtype == np.uint8
        assert image.std() > 5  # not a blank frame
    env.close()


# ---- cameras as the environment delivers them ------------------------------------------------------------


def _rendering_env(**config: object) -> YamStationEnv:
    env = YamStationEnv(EnvConfig(cameras=tuple(scene.load_cameras()), **config))
    try:
        env.render("overhead")
    except Exception as error:
        pytest.skip(f"no OpenGL context for offscreen rendering: {error}")
    return env


def test_camera_frames_are_as_stale_as_the_real_ones() -> None:
    """With latency on, the frame handed out with a state shows the arm where it was a moment earlier."""
    start = np.array([0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 1.0])
    swing = np.array([0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 1.0])  # base joint: the whole arm sweeps across the image
    frames = {}
    for latency in (True, False):
        env = _rendering_env(camera_latency=latency)
        env.reset(start)
        env._renderers["overhead"].image_model.noise_sigma = 0.0
        for _ in range(8):
            observation = env.step(swing)
        frames[latency] = (
            observation[IMAGE_KEY.format("overhead")].astype(float),
            env.render("overhead", noise=False).astype(float),
        )
        env.close()
    delayed, now = frames[True]
    fresh, _ = frames[False]
    assert np.abs(fresh - now).mean() < 1.0  # without latency the frame is the present
    assert np.abs(delayed - now).mean() > 2.0 * np.abs(fresh - now).mean() + 0.5  # with it, the arm has moved on since


def test_appearance_randomisation_is_reversible() -> None:
    env = _rendering_env()
    env.reset()
    calibrated = env.render("overhead", noise=False).astype(float)
    env.randomize_appearance(np.random.default_rng(3))
    assert np.abs(env.render("overhead", noise=False).astype(float) - calibrated).mean() > 1.0
    env.randomize_appearance(None)
    assert np.abs(env.render("overhead", noise=False).astype(float) - calibrated).mean() < 1e-6
    env.close()


def test_held_ball_is_drawn_squeezed(env: YamStationEnv) -> None:
    env.reset(np.array([0, 0, 0, 0, 0, 0, 0.4]))
    env.holding_ball = True
    env._shape_ball()
    size = env.model.geom_size[env.model.geom("ball_visual").id]
    radius = scene.object_parameters()["ball"]["radius"]
    assert size[0] == pytest.approx(0.4 * scene.FINGER_TRAVEL, abs=1e-6)  # as thick as the finger gap
    assert size[1] > radius  # and bulging across it
    env.holding_ball = False
    env._shape_ball()
    assert np.allclose(env.model.geom_size[env.model.geom("ball_visual").id], radius)


# ---- against the recordings (skipped where the dataset is not available) ------------------------------------


def test_recorded_episode_replays_to_the_same_outcome() -> None:
    """A recorded demonstration, replayed open-loop through the physics, puts the ball on the same cup."""
    from i2rt.simulation import episodes as episode_data
    from i2rt.simulation.calibration.common import DEFAULT_DATASET
    from i2rt.simulation.replay import replay_physics

    if not (DEFAULT_DATASET / "episode_0020" / "meta.json").is_file():
        pytest.skip("recorded dataset not present")
    episode = episode_data.load()["episode_0020"]
    setup = episode_data.episode_setup(episode)
    result = replay_physics(YamStationEnv(EnvConfig(cameras=())), episode, setup)
    assert result.success
    tracking = np.sqrt(((result.sim_states - episode.state) ** 2).mean(axis=0))
    assert tracking[:6].max() < np.radians(1.0)  # simulated joints within a degree of the real ones, RMS
