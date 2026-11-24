"""Fit the simulated arm's joint mechanics to how the real arm followed its commands.

Every recorded frame holds the command sent to the follower (``action``) and where the follower
actually was (``state``). Replaying the commands through the simulated motors and comparing the
resulting joint angles with the recorded ones identifies, per joint:

* ``frictionloss`` -- dry friction. It is what lets the real joints settle up to ~0.015 rad short of
  their targets and stay there;
* ``damping`` and ``armature`` -- viscous loss and reflected rotor inertia, which set the lag;
* an effective ``gravity_comp_factor`` (pitch joints only). The real controller scales its gravity
  feed-forward by fixed factors, but what reaches the joint also depends on things the controller's
  model lacks -- the wrist camera and its cable, torque-constant error. One number per joint, chosen
  so the simulated arm rests off its targets the way the real one does, stands in for all of it.

The motor gains (kp, kd) are the real ones and are not fitted. Joints are fitted one at a time, in
parallel, since each mostly affects its own tracking; a second pass picks up the coupling.

    python -m i2rt.simulation.calibration.dynamics
"""

from __future__ import annotations

import json
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from scipy.optimize import minimize

from i2rt.simulation.calibration.common import CALIB_DIR, DEFAULT_DATASET, load_episodes

PITCH_JOINTS = (1, 2, 3)  # joints 2-4: the ones that carry gravity


@dataclass
class Args:
    dataset_root: Path = DEFAULT_DATASET
    every: int = 3
    """Use every n-th episode."""
    passes: int = 2
    iters: int = 90
    out: Path = CALIB_DIR / "dynamics.json"


def _make_env():  # noqa: ANN202 - imported lazily so worker processes build their own
    from i2rt.simulation import scene
    from i2rt.simulation.env import EnvConfig, YamStationEnv

    return YamStationEnv(EnvConfig(cameras=(), scene=scene.SceneOptions(backdrop=False, include_objects=False)))


def simulate(env, episodes: list) -> list[np.ndarray]:  # noqa: ANN001
    """Simulated joint states at the recorded frame times, for each episode."""
    out = []
    for episode in episodes:
        env.reset(episode.state[0])
        states = [env.joint_state()]
        for k in range(1, len(episode)):
            env.advance(episode.action[k], duration=float(episode.timestamp[k] - episode.timestamp[k - 1]))
            states.append(env.joint_state())
        out.append(np.array(states))
    return out


def tracking_error(env, episodes: list) -> np.ndarray:  # noqa: ANN001
    """RMS of (simulated - recorded) joint angle per joint, over all frames."""
    diffs = np.concatenate([sim - ep.state for sim, ep in zip(simulate(env, episodes), episodes, strict=True)])
    return np.sqrt((diffs**2).mean(axis=0))


def _fit_joint(job: tuple) -> tuple[int, dict, float]:
    """Worker: fit one joint's parameters with the others held at ``params``."""
    joint, params, dataset_root, every, iters = job
    episodes = load_episodes(dataset_root)[::every]
    env = _make_env()
    fit_gravity = joint in PITCH_JOINTS
    start = [
        np.log(params["frictionloss"][joint]),
        np.log(params["damping"][joint]),
        np.log(params["armature"][joint]),
    ]
    if fit_gravity:
        start.append(params["gravity_comp_factor"][joint])

    def apply(x: np.ndarray) -> dict:
        trial = json.loads(json.dumps(params))
        trial["frictionloss"][joint] = float(np.exp(x[0]))
        trial["damping"][joint] = float(np.exp(x[1]))
        trial["armature"][joint] = float(np.exp(x[2]))
        if fit_gravity:
            trial["gravity_comp_factor"][joint] = float(x[3])
        return trial

    def loss(x: np.ndarray) -> float:
        if np.abs(x[:3] - start[:3]).max() > 4.0:  # keep the search within e^4 of the starting values
            return 1.0
        env.set_dynamics(apply(x))
        return float(tracking_error(env, episodes)[joint])

    steps = [0.5, 0.5, 0.5] + ([0.1] if fit_gravity else [])
    result = minimize(
        loss,
        np.array(start),
        method="Nelder-Mead",
        options={
            "maxiter": iters,
            "xatol": 1e-3,
            "fatol": 1e-6,
            "initial_simplex": np.vstack([start, np.array(start) + np.diag(steps)]),
        },
    )
    return joint, apply(result.x), float(result.fun)


def main(args: Args) -> None:
    from i2rt.simulation import scene

    t0 = time.time()
    params = {k: (list(v) if isinstance(v, list) else v) for k, v in scene.DEFAULT_DYNAMICS.items()}
    episodes = load_episodes(args.dataset_root)[:: args.every]
    env = _make_env()
    env.set_dynamics(params)
    print(
        f"{len(episodes)} episodes; starting tracking error rms [rad]: {[round(float(v), 4) for v in tracking_error(env, episodes)]}",
        flush=True,
    )
    with multiprocessing.get_context("spawn").Pool(6) as pool:
        for sweep in range(args.passes):
            jobs = [(joint, params, args.dataset_root, args.every, args.iters) for joint in range(6)]
            for joint, fitted, _ in pool.map(_fit_joint, jobs):
                for key in ("frictionloss", "damping", "armature", "gravity_comp_factor"):
                    params[key][joint] = fitted[key][joint]
            env.set_dynamics(params)
            print(
                f"pass {sweep + 1}: tracking error rms [rad]: {[round(float(v), 4) for v in tracking_error(env, episodes)]} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    params["notes"] = "arm joint parameters fitted by calibration/dynamics.py; kp/kd are the real motor gains"
    args.out.write_text(json.dumps(params, indent=2) + "\n")
    for key in ("frictionloss", "damping", "armature", "gravity_comp_factor"):
        print(f"  {key}: {np.round(params[key], 4).tolist()}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
