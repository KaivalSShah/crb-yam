"""Turn recorded demonstration episodes into a LeRobot dataset, ready for ``train_lerobot``.

Reads the ``episode_NNNN`` folders that ``teleop_leader_follower --demo-root`` writes (see
``episode_recorder``) and adds each as one dataset episode. This is the slow step -- every episode's
camera frames are encoded into video, roughly 1-2 s per second of footage -- which is why recording
leaves it until now. Requires LeRobot (``uv pip install lerobot``).

Re-running it after recording more only adds the new episodes: the episodes already in a dataset
are listed in ``<demo root>/converted.json``.

    python -m i2rt.robot_learning.convert_to_lerobot --demo-root data/yam_demos --dataset-root data/yam_lerobot
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro

from i2rt.robot_learning.episode_recorder import CAMERAS, episode_dirs


@dataclass
class Args:
    demo_root: str
    """Folder passed to teleop as ``--demo-root``: one ``episode_NNNN`` folder per demonstration."""
    dataset_root: str
    """LeRobot dataset to create, or to add the new episodes to."""
    dataset_repo_id: str = "i2rt/yam_teleop"
    """Repo id stored in the dataset. Pass the same value to ``train_lerobot``."""


def _load_ledger(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text()) if path.is_file() else {}


def main(args: Args) -> None:
    demo_root = Path(args.demo_root)
    dataset_root = Path(args.dataset_root)
    episodes = episode_dirs(demo_root)
    if not episodes:
        raise SystemExit(f"no complete episodes in {demo_root}")

    ledger_path = demo_root / "converted.json"
    ledger = _load_ledger(ledger_path)
    key = str(dataset_root.resolve())
    done = set(ledger.get(key, []))
    if dataset_root.exists() and key not in ledger and any(dataset_root.iterdir()):
        raise SystemExit(f"{dataset_root} already exists and was not made from {demo_root}; pick a new --dataset-root")
    todo = [e for e in episodes if e.name not in done]
    if not todo:
        print(f"All {len(episodes)} episodes are already in {dataset_root}.")
        return

    metas = {e: json.loads((e / "meta.json").read_text()) for e in todo}
    first = metas[todo[0]]
    for episode, meta in metas.items():
        for field in ("fps", "joint_names", "cameras"):
            if meta[field] != first[field]:
                raise SystemExit(f"{episode.name} has a different {field} from {todo[0].name}; convert them apart")
    if list(first["cameras"]) != list(CAMERAS):
        raise SystemExit(f"episodes record cameras {first['cameras']}; this converter expects {list(CAMERAS)}")

    from i2rt.robot_learning.lerobot_recorder import TeleopDataset

    dataset = TeleopDataset(
        root=dataset_root,
        repo_id=args.dataset_repo_id,
        fps=first["fps"],
        names=first["joint_names"],
        task=first["task"],
        robot_type=first["robot_type"],
    )
    try:
        for n, episode in enumerate(todo, 1):
            meta = metas[episode]
            states = np.load(episode / "state.npy")
            actions = np.load(episode / "action.npy")
            print(f"[{n}/{len(todo)}] {episode.name}: {len(states)} frames, task '{meta['task']}'", flush=True)
            skipped = 0
            for i in range(len(states)):
                images = [cv2.imread(str(episode / camera / f"{i:06d}.jpg")) for camera in CAMERAS]
                if any(image is None for image in images):
                    skipped += 1
                    continue
                dataset.add(states[i], actions[i], *images, task=meta["task"])
            if skipped:
                print(f"    {skipped} frame(s) had a missing image and were left out", flush=True)
            dataset.save_episode()
            # Ledger after each episode, so an interrupted run resumes where it stopped.
            ledger.setdefault(key, []).append(episode.name)
            ledger_path.write_text(json.dumps(ledger, indent=2))
    finally:
        dataset.close()
    print(f"Done: {dataset_root} now holds {len(ledger[key])} episode(s).")


if __name__ == "__main__":
    main(tyro.cli(Args))
