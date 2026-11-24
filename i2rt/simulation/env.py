"""The simulated station as an environment with the real robot's interface.

Observations and actions use the dataset's conventions, so a policy trained on the recorded
demonstrations runs here unchanged and sim rollouts can be recorded in the same format:

* action: 7 floats -- six joint position targets [rad] and the gripper opening in [0, 1] (1 = open)
* ``observation.state``: 7 floats -- the six measured joint angles and the measured gripper opening
* ``observation.images.overhead`` / ``observation.images.right_wrist``: 480x640x3 uint8 RGB

One ``step`` is one control period of the recorded data (1/30 s by default). Inside it the physics
runs at 500 Hz with the real arm's control law: each motor's own PD loop toward the target plus a
gravity feed-forward computed from the arm model, scaled by the per-joint factors the real arm uses.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np

from i2rt.simulation import scene
from i2rt.simulation.rendering import CameraRenderer, make_renderers

STATE_KEY = "observation.state"
IMAGE_KEY = "observation.images.{}"
HOME = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
COMMAND_LIMIT_BUFFER = 0.15  # rad beyond a joint limit that i2rt.robots.get_robot lets a position command reach


@dataclass
class EnvConfig:
    control_hz: float = 30.0
    """Rate at which actions arrive (the dataset's frame rate)."""
    soft_ball_grasp: bool = True
    """Hold the ball in the gripper while both fingers squeeze it.

    The real ball is a soft shell: pinched anywhere between the fingertips it dents and stays put.
    A rigid sphere pinched off-centre is squeezed out instead, however soft its contacts. So while
    both fingers press into the ball it is tied to the gripper where it is, and let go the moment
    the fingers open wider than the ball. Off leaves the grasp to rigid-body contact alone."""
    camera_latency: bool = True
    """Deliver each camera's frame as old as the real camera's are (``ImageModel.latency``: about
    80 ms overhead, 55 ms wrist). The real recordings pair every joint reading with frames that stale,
    so a policy trained on them expects it. Off pairs each state with a frame of the same instant."""
    interpolate_targets: bool = True
    """Ramp each motor target from the previous action to the new one over the step, as the real
    100 Hz command stream between two recorded frames does. Off holds the new target for the step."""
    cameras: tuple[str, ...] = ("overhead", "right_wrist")
    supersample: int = 2
    scene: scene.SceneOptions = field(default_factory=scene.SceneOptions)


class YamStationEnv:
    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self.model = scene.build_model(self.config.scene)
        self.data = mujoco.MjData(self.model)
        self.dynamics = scene.dynamics_parameters()
        self.table = self.config.scene.table_plane()
        self._arm_qpos = np.array([self.model.joint(j).qposadr[0] for j in scene.ARM_JOINTS])
        self._arm_dof = np.array([self.model.joint(j).dofadr[0] for j in scene.ARM_JOINTS])
        self._finger_qpos = np.array([self.model.joint(j).qposadr[0] for j in scene.FINGER_JOINTS])
        self._finger_dof = np.array([self.model.joint(j).dofadr[0] for j in scene.FINGER_JOINTS])
        self._arm_act = np.array([self.model.actuator(f"motor{i + 1}").id for i in range(6)])
        # The real controller clips arm targets to the model's joint limits widened by a buffer
        # (get_yam_robot's "safety buffer") before they reach the motors; the motors never see more.
        arm_range = np.array([self.model.joint(name).range for name in scene.ARM_JOINTS])
        self._command_range = (arm_range[:, 0] - COMMAND_LIMIT_BUFFER, arm_range[:, 1] + COMMAND_LIMIT_BUFFER)
        self._finger_act = np.array([self.model.actuator(f"finger_{j}").id for j in scene.FINGER_JOINTS])
        self._gravity_factor = np.asarray(self.dynamics["gravity_comp_factor"], dtype=float)
        # The real controller computes gravity torques from the plain arm + gripper model: it knows
        # nothing of the wrist camera or of whatever the gripper is holding, and neither does this.
        self._gravity_model = scene.build_gravity_model()
        self._gravity_data = mujoco.MjData(self._gravity_model)
        self._renderers: dict[str, CameraRenderer] = {}
        self._frames: dict[str, collections.deque] = {}  # per camera: the latest frames, oldest first
        self._frame_delay: dict[
            str, tuple[int, int]
        ] = {}  # per camera: latency as (substeps into a period, whole periods)
        self._target = HOME.copy()
        self._substeps = max(1, round(1.0 / (self.config.control_hz * self.model.opt.timestep)))
        self._free_joints = {}
        self.holding_ball = False
        self._hold_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, scene.BALL_HOLD)
        if self._hold_id >= 0:
            self._ball_geom = self.model.geom("ball").id
            self._finger_geoms = tuple(self.model.geom(name).id for name in scene.FINGER_COLLIDERS)
            self._ball_diameter = 2.0 * float(self.model.geom_size[self._ball_geom, 0])
            self._ball_visual = self.model.geom("ball_visual").id
            # Compiled as "same frame as its body" (a shortcut that ignores the geom's own
            # orientation); it gets one of its own whenever the ball is squeezed.
            self.model.geom_sameframe[self._ball_visual] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        if self.config.scene.include_objects:
            self._free_joints = {name: self.model.joint(f"{name}_free").qposadr[0] for name in scene.OBJECT_NAMES}
        self.reset()

    def set_dynamics(self, params: dict) -> None:
        """Change joint mechanics on the live model (used by ``calibration/dynamics.py``)."""
        self.dynamics.update(params)
        dyn = self.dynamics
        self.model.dof_armature[self._arm_dof] = dyn["armature"]
        self.model.dof_damping[self._arm_dof] = dyn["damping"]
        self.model.dof_frictionloss[self._arm_dof] = dyn["frictionloss"]
        self.model.dof_armature[self._finger_dof] = dyn["finger_armature"]
        self.model.dof_damping[self._finger_dof] = dyn["finger_damping"]
        self.model.actuator_gainprm[self._finger_act, 0] = dyn["finger_kp"]
        self.model.actuator_biasprm[self._finger_act, 1] = -dyn["finger_kp"]
        self.model.actuator_biasprm[self._finger_act, 2] = -dyn["finger_kv"]
        self.model.actuator_forcerange[self._finger_act] = [-dyn["finger_force"], dyn["finger_force"]]
        self._gravity_factor = np.asarray(dyn["gravity_comp_factor"], dtype=float)
        mujoco.mj_setConst(self.model, self.data)

    # ---- state -------------------------------------------------------------------------------------
    def reset(
        self, joint_state: np.ndarray | None = None, objects: dict[str, scene.ObjectState] | None = None
    ) -> dict:
        """Put the arm at ``joint_state`` (7 values, dataset convention) and the objects on the table."""
        mujoco.mj_resetData(self.model, self.data)
        self.holding_ball = False
        self._frames.clear()
        state = HOME if joint_state is None else np.asarray(joint_state, dtype=float)
        self.data.qpos[self._arm_qpos] = state[:6]
        self.data.qpos[self._finger_qpos] = state[6] * scene.FINGER_TRAVEL
        params = scene.object_parameters()
        states = {**scene.default_object_states(), **self.config.scene.objects, **(objects or {})}
        for name, adr in self._free_joints.items():
            pos, quat = scene.object_pose(name, states[name], params, self.table)
            self.data.qpos[adr : adr + 3] = pos
            self.data.qpos[adr + 3 : adr + 7] = quat
        self._target = state.copy()
        self._write_targets(self._target)
        mujoco.mj_forward(self.model, self.data)
        self._apply_gravity_feedforward()
        return self.observe()

    def joint_state(self) -> np.ndarray:
        """Measured joints in the dataset's convention: six angles and the gripper opening in [0, 1]."""
        gripper = self.data.qpos[self._finger_qpos].mean() / scene.FINGER_TRAVEL
        return np.concatenate([self.data.qpos[self._arm_qpos], [gripper]]).astype(np.float32)

    def object_position(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    # ---- control ------------------------------------------------------------------------------------
    def _write_targets(self, target: np.ndarray) -> None:
        self.data.ctrl[self._arm_act] = np.clip(target[:6], self._command_range[0], self._command_range[1])
        self.data.ctrl[self._finger_act] = np.clip(target[6], 0.0, 1.0) * scene.FINGER_TRAVEL

    def _apply_gravity_feedforward(self) -> None:
        gd = self._gravity_data
        gd.qpos[:6] = self.data.qpos[self._arm_qpos]
        gd.qpos[6:8] = self.data.qpos[self._finger_qpos]
        gd.qvel[:] = 0.0
        mujoco.mj_forward(self._gravity_model, gd)
        self.data.qfrc_applied[self._arm_dof] = gd.qfrc_bias[:6] * self._gravity_factor

    def step(self, action: np.ndarray, duration: float | None = None) -> dict:
        """Apply one action for one control period (or ``duration`` seconds) and return the observation.

        The joint state in the observation is the one at the end of the step; each camera frame is
        the one that camera would have delivered with it, i.e. exposed ``latency`` earlier.
        """
        self.advance(action, duration, capture=True)
        observation: dict = {STATE_KEY: self.joint_state()}
        for camera, frames in self._frames.items():
            delay = self._frame_delay[camera][1]
            observation[IMAGE_KEY.format(camera)] = frames[max(len(frames) - 1 - delay, 0)]
        return observation

    def advance(self, action: np.ndarray, duration: float | None = None, capture: bool = False) -> None:
        """Run the physics for one control period toward ``action``.

        Without ``capture`` nothing is rendered: for dynamics work, where images would only cost time.
        With it, each camera takes its frame at the moment within the period that its latency calls for.
        """
        action = np.asarray(action, dtype=float)
        substeps = self._substeps if duration is None else max(1, round(duration / self.model.opt.timestep))
        previous = self._target
        capture_at = {}
        if capture:
            self._ensure_frames()
            # A frame ``lag`` substeps old at the end of a period was exposed ``lag % period`` substeps
            # before the end of some earlier period; taking one frame per period at that offset and
            # handing out the one from ``lag // period`` periods ago gives exactly that.
            capture_at = {camera: max(substeps - remainder, 1) for camera, (remainder, _) in self._frame_delay.items()}
        for i in range(substeps):
            blend = (i + 1) / substeps if self.config.interpolate_targets else 1.0
            self._write_targets(previous + blend * (action - previous))
            if i % 2 == 0:  # the real control loop refreshes its feed-forward at ~250 Hz
                self._apply_gravity_feedforward()
            mujoco.mj_step(self.model, self.data)
            if self._hold_id >= 0 and self.config.soft_ball_grasp:
                self._update_ball_hold()
            for camera, index in capture_at.items():
                if index == i + 1:
                    self._frames[camera].append(self.render(camera))
        self._target = action.copy()

    def _ensure_frames(self) -> None:
        """Start each camera's frame history with the present view (after a reset there is no past)."""
        if self._frames:
            return
        for camera in self.config.cameras:
            if camera not in scene.load_cameras():
                continue
            if not self._renderers:
                self._renderers = make_renderers(self.model, self.config.supersample)
            lag = (
                round(self._renderers[camera].image_model.latency / self.model.opt.timestep)
                if self.config.camera_latency
                else 0
            )
            self._frame_delay[camera] = (lag % self._substeps, lag // self._substeps)
            self._frames[camera] = collections.deque([self.render(camera)], maxlen=lag // self._substeps + 2)

    def _update_ball_hold(self) -> None:
        """Tie the ball to the gripper while both fingers press into it; release it when they open."""
        data = self.data
        finger_gap = float(data.qpos[self._finger_qpos].sum())  # pad-to-pad distance: zero when closed
        if self.holding_ball:
            if finger_gap >= self._ball_diameter - 0.0005:
                data.eq_active[self._hold_id] = 0
                self.holding_ball = False
            return
        if finger_gap >= self._ball_diameter - 0.001:
            return
        pressing = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.dist > -0.0005:
                continue
            if contact.geom1 == self._ball_geom and contact.geom2 in self._finger_geoms:
                pressing.add(contact.geom2)
            elif contact.geom2 == self._ball_geom and contact.geom1 in self._finger_geoms:
                pressing.add(contact.geom1)
        if len(pressing) == 2:
            gripper = data.body("gripper")
            ball = data.body("ball").xpos
            self.model.eq_data[self._hold_id, 0:3] = gripper.xmat.reshape(3, 3).T @ (ball - gripper.xpos)
            self.model.eq_data[self._hold_id, 3:6] = 0.0
            data.eq_active[self._hold_id] = 1
            self.holding_ball = True

    # ---- observation -------------------------------------------------------------------------------
    def _shape_ball(self) -> None:
        """Draw the ball as it is right now: round when free, flattened between the fingers when held.

        Collision keeps using the sphere; only the picture changes. Held, the ball is as thick as the
        finger gap along the closing direction and bulges a little across it.
        """
        if self._hold_id < 0:
            return
        radius = self._ball_diameter / 2.0
        size, quat = np.full(3, radius), np.array([1.0, 0.0, 0.0, 0.0])
        if self.holding_ball:
            half_gap = float(np.clip(self.data.qpos[self._finger_qpos].sum() / 2.0, 0.4 * radius, radius))
            bulge = radius * (radius / half_gap) ** 0.25
            size = np.array([half_gap, bulge, bulge])
            # Orient the ellipsoid's short axis along the gripper's closing direction (its +X), whatever
            # way the ball body itself has rolled.
            ball_to_gripper = np.zeros(4)
            inverse = np.zeros(4)
            mujoco.mju_negQuat(inverse, self.data.body("ball").xquat)
            mujoco.mju_mulQuat(ball_to_gripper, inverse, self.data.body("gripper").xquat)
            quat = ball_to_gripper
        if not (
            np.allclose(self.model.geom_size[self._ball_visual], size)
            and np.allclose(self.model.geom_quat[self._ball_visual], quat)
        ):
            self.model.geom_size[self._ball_visual] = size
            self.model.geom_quat[self._ball_visual] = quat
            mujoco.mj_kinematics(self.model, self.data)

    def render(self, camera: str, noise: bool = True) -> np.ndarray:
        """What ``camera`` records right now (RGB). ``noise=False`` leaves out sensor noise and flicker."""
        if not self._renderers:
            self._renderers = make_renderers(self.model, self.config.supersample)
        self._shape_ball()
        return self._renderers[camera].render(self.data, noise=noise)

    def randomize_appearance(self, rng: np.random.Generator | None, **limits: float) -> None:
        """Jitter both cameras' poses, paints and exposure for domain randomisation; ``None`` restores the calibration.

        The calibration has residuals of a few millimetres and a few grey levels; training or
        evaluating across perturbations of that size keeps a policy from leaning on the exact values.
        ``limits`` are passed to ``CameraRenderer.randomize`` (``shift``, ``tilt``, ``colour``, ``exposure``).
        """
        if not self._renderers:
            self._renderers = make_renderers(self.model, self.config.supersample)
        for renderer in self._renderers.values():
            renderer.randomize(rng, **limits)
        mujoco.mj_forward(self.model, self.data)  # camera poses live in data

    def set_backdrop(self, episode: str | None) -> None:
        """Show the room around the table as it was during a recorded episode (``None``: as it usually was).

        Chairs and people moved between episodes. ``bake_overhead.py`` keeps one picture of the room
        per episode for the overhead camera, and ``bake_wrist.py`` one set of wall and floor textures
        per episode for the wrist camera; this swaps them in.
        """
        if not self.config.cameras:
            return  # nothing is rendered: nothing to swap
        if not self._renderers:
            self._renderers = make_renderers(self.model, self.config.supersample)
        backdrops = scene.ASSETS / "backdrops"
        if "overhead" in self._renderers:
            path = backdrops / "overhead" / f"{episode}.jpg"
            if episode is None or not path.is_file():
                path = scene.ASSETS / "textures" / "overhead_backdrop.png"
            self._renderers["overhead"].set_texture(
                "overhead_backdrop_tex", cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            )
        if "right_wrist" in self._renderers:
            for sheet in scene.load_json("wrist_layout.json", default={}).get("sheets", []):
                path = backdrops / "right_wrist" / f"{episode}_{sheet['name']}.jpg"
                if episode is None or not path.is_file():
                    path = scene.ASSETS / sheet["texture"]
                if sheet["name"] != "wrist_table":
                    self._renderers["right_wrist"].set_texture(
                        f"{sheet['name']}_tex", cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                    )

    def observe(self, images: bool = True) -> dict:
        """The state and both cameras' views as of this instant (no camera latency: there is no past to show)."""
        obs: dict = {STATE_KEY: self.joint_state()}
        if images:
            for camera in self.config.cameras:
                if camera in scene.load_cameras():
                    obs[IMAGE_KEY.format(camera)] = self.render(camera)
        return obs

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers = {}
