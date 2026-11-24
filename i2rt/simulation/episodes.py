"""Recorded episodes as simulation set-ups: where the cups and the ball were when each one started."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from i2rt.simulation import scene
from i2rt.simulation.calibration.common import DEFAULT_DATASET, Episode, load_episodes, set_robot_state
from i2rt.simulation.table import TablePlane

GRIPPER_CLOSING = 0.75  # gripper command below this means the operator is closing on the ball


@dataclass
class EpisodeSetup:
    """Initial conditions of one recorded episode, plus what the operator did with the ball."""

    objects: dict[str, scene.ObjectState]
    source_cup: str | None  # the cup the ball starts on
    target_cup: str | None  # the cup it was put on
    grasp_frame: int | None  # first frame with the fingers closed on the ball
    release_frame: int | None  # last frame before the fingers open again


def episode_setup(episode: Episode, objects: dict | None = None, table: TablePlane | None = None) -> EpisodeSetup:
    """Where things were at the start of ``episode``.

    Cup positions come from the overhead camera (``calibration/objects.py``). Which cup carried the
    ball comes from the arm itself: the cup nearest the fingers when they closed is the source, the
    one nearest when they opened is the target.
    """
    objects = objects if objects is not None else scene.load_json("objects.json")
    # Object positions were measured per episode of one dataset; an episode of the same name from
    # another dataset has nothing to do with them.
    same_dataset = objects.get("dataset") in (None, episode.path.resolve().parent.name)
    fitted = objects["episodes"].get(episode.name, {}) if same_dataset else {}
    table = table or TablePlane.load()
    overhead = scene.load_cameras()["overhead"]
    # A cup is recorded as the pixel its base appears at; where that is on the table depends on the
    # table plane, so it is resolved here rather than stored.
    located = {
        name: overhead.pixel_to_table(np.array([entry["base_px"]]), table)[0, :2]
        for name, entry in fitted.items()
        if entry
    }
    cups = {name: xy for name, xy in located.items() if name != "box"}
    states = {
        name: scene.ObjectState(xy=(float(xy[0]), float(xy[1])), yaw=float(fitted[name].get("yaw", 0.0)))
        for name, xy in located.items()
    }
    for name in ("cup_orange", "cup_blue"):
        states.setdefault(
            name, scene.ObjectState(xy=(1.5, 1.2 if name == "cup_blue" else 1.0))
        )  # not seen: off to the side
    states.setdefault("box", scene.default_object_states()["box"])

    closed = np.flatnonzero(episode.action[:, 6] < GRIPPER_CLOSING)
    if len(closed) < 5 or not cups:
        states["ball"] = scene.ObjectState(xy=(1.5, 0.8))
        return EpisodeSetup(states, None, None, None, None)
    grasp, release = int(closed[0]) + 6, int(closed[-1])
    model = scene.build_gravity_model()
    data = mujoco.MjData(model)
    site = model.site("grasp_site").id
    at = {}
    for label, frame in (("grasp", grasp), ("release", release)):
        set_robot_state(model, data, episode.state[min(frame, len(episode) - 1)])
        at[label] = data.site_xpos[site][:2].copy()
    source = min(cups, key=lambda name: np.linalg.norm(cups[name] - at["grasp"]))
    target = min(cups, key=lambda name: np.linalg.norm(cups[name] - at["release"]))
    states["ball"] = scene.ObjectState(xy=states[source].xy, z=scene.ball_rest_height())
    return EpisodeSetup(states, source, target, grasp, release)


def load(dataset_root: Path = DEFAULT_DATASET) -> dict[str, Episode]:
    return {episode.name: episode for episode in load_episodes(dataset_root)}


def all_setups(dataset_root: Path = DEFAULT_DATASET) -> dict[str, EpisodeSetup]:
    objects = json.loads((scene.RESULTS / "objects.json").read_text())
    return {name: episode_setup(episode, objects) for name, episode in load(dataset_root).items()}


STARTS_FILE = scene.RESULTS / "episode_starts.json"


def save_starts(dataset_root: Path = DEFAULT_DATASET, path: Path = STARTS_FILE) -> int:
    """Write every recorded episode's starting condition to ``path``; returns how many were written.

    Starting a rollout "as a recorded episode began" then no longer needs the recordings themselves:
    the file travels with the package (to a training server, say) while the dataset stays put.
    """
    starts = {}
    for name, episode in load(dataset_root).items():
        setup = episode_setup(episode)
        if setup.target_cup is None:
            continue
        starts[name] = {
            "state": [float(v) for v in episode.state[0]],
            "objects": {
                key: {"xy": list(value.xy), "yaw": value.yaw, "z": value.z} for key, value in setup.objects.items()
            },
            "source_cup": setup.source_cup,
            "target_cup": setup.target_cup,
        }
    path.write_text(json.dumps(starts, indent=1) + "\n")
    return len(starts)


def load_starts(path: Path = STARTS_FILE) -> dict[str, tuple[np.ndarray, EpisodeSetup]]:
    """Starting conditions of the recorded episodes: name -> (arm joint state, object set-up)."""
    starts = {}
    for name, entry in json.loads(path.read_text()).items():
        objects = {
            key: scene.ObjectState(xy=tuple(value["xy"]), yaw=value["yaw"], z=value["z"])
            for key, value in entry["objects"].items()
        }
        starts[name] = (
            np.array(entry["state"]),
            EpisodeSetup(objects, entry["source_cup"], entry["target_cup"], None, None),
        )
    return starts


def sample_start(
    rng: np.random.Generator, jitter: float = 0.0, episode: str | None = None
) -> tuple[str, np.ndarray, EpisodeSetup]:
    """A starting condition from the stored recorded ones, optionally with the cups nudged: (name, arm state, set-up).

    Cups move by up to ``jitter`` metres (the cup under the open gripper by at most 12 mm, the room the
    gripper leaves around the ball), which stays inside what the demonstrations cover without repeating
    a recorded start exactly.
    """
    starts = load_starts()
    name = episode or sorted(starts)[int(rng.integers(len(starts)))]
    state, setup = starts[name]
    if jitter <= 0:
        return name, state, setup
    shift = {cup: rng.uniform(-jitter, jitter, size=2) for cup in ("cup_orange", "cup_blue")}
    shift[setup.source_cup] = np.clip(shift[setup.source_cup], -0.012, 0.012)
    objects = dict(setup.objects)
    for cup, delta in shift.items():
        placed = objects[cup]
        objects[cup] = scene.ObjectState(
            xy=(placed.xy[0] + delta[0], placed.xy[1] + delta[1]), yaw=placed.yaw, z=placed.z
        )
    ball, moved = objects["ball"], shift[setup.source_cup]  # the ball rides on its cup
    objects["ball"] = scene.ObjectState(xy=(ball.xy[0] + moved[0], ball.xy[1] + moved[1]), yaw=ball.yaw, z=ball.z)
    return name, state, EpisodeSetup(objects, setup.source_cup, setup.target_cup, None, None)
