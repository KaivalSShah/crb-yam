"""Open the simulated station in MuJoCo's interactive viewer.

On macOS the viewer needs ``mjpython`` (it ships with the mujoco package)::

    .venv/bin/mjpython -m i2rt.simulation.view            # Linux: python -m i2rt.simulation.view

The arm holds its pose under the simulated controller; drag bodies with the mouse (double-click,
then ctrl + right-drag) to push things around. Press ``]`` / ``[`` in the viewer to look through
the calibrated cameras (``overhead``, ``right_wrist``). The photographed backdrops belong to one
camera each; ``--backdrop`` picks whose is shown.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np
import tyro

from i2rt.simulation import episodes as episode_data
from i2rt.simulation import scene
from i2rt.simulation.env import EnvConfig, YamStationEnv


@dataclass
class Args:
    episode: str | None = None
    """Start from this recorded episode's first frame (arm pose and object positions) and replay its actions."""
    backdrop: Literal["overhead", "right_wrist"] = "right_wrist"
    """Whose photographed surroundings to show (the wrist camera's are real 3D surfaces)."""
    loop: bool = True


def main(args: Args) -> None:
    env = YamStationEnv(EnvConfig(cameras=()))
    episode = setup = None
    if args.episode is not None:
        episode = episode_data.load()[args.episode]
        setup = episode_data.episode_setup(episode)
        env.reset(episode.state[0], objects=setup.objects)
    target = env.joint_state().astype(float)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Both per-camera groups are shown (the wrist-camera mount lives in one of them), minus the
        # photographed sheets that belong to the other camera.
        other = "overhead_" if args.backdrop == "right_wrist" else "wrist_"
        for geom in range(env.model.ngeom):
            if env.model.geom(geom).name.startswith(other):
                env.model.geom_group[geom] = scene.GROUP_COLLISION
        viewer.opt.geomgroup[scene.GROUP_OVERHEAD_ONLY] = 1
        viewer.opt.geomgroup[scene.GROUP_WRIST_ONLY] = 1
        viewer.opt.geomgroup[scene.GROUP_COLLISION] = 0
        k = 0
        while viewer.is_running():
            start = time.time()
            if episode is not None:
                k += 1
                if k >= len(episode):
                    if not args.loop:
                        break
                    k = 0
                    env.reset(episode.state[0], objects=setup.objects)
                target = episode.action[k]
            env.advance(np.asarray(target))
            viewer.sync()
            time.sleep(max(0.0, 1.0 / env.config.control_hz - (time.time() - start)))


if __name__ == "__main__":
    main(tyro.cli(Args))
