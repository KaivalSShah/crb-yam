"""Train an ACT policy on a dataset made by convert_to_lerobot from teleop recordings.

Requires LeRobot (``uv pip install lerobot``). The policy reads the recorded
``observation.state``, ``action``, ``observation.images.overhead`` and
``observation.images.right_wrist`` features and adapts to them.

    python -m i2rt.robot_learning.train_lerobot --dataset-root datasets/yam_teleop
"""

import shutil
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    dataset_root: str
    """LeRobot dataset made by ``convert_to_lerobot`` (its ``--dataset-root``)."""
    dataset_repo_id: str = "i2rt/yam_teleop"
    """Same repo id passed to ``convert_to_lerobot``."""
    output_dir: str = "outputs/train/yam_act"
    policy: str = "act"
    """LeRobot policy type. ``act`` is the usual first policy for this dataset."""
    device: str = "mps"
    """``mps`` on Apple silicon, ``cuda`` on NVIDIA, ``cpu`` otherwise."""
    batch_size: int = 8
    steps: int = 20000


def main(args: Args) -> None:
    if shutil.which("lerobot-train") is None:
        raise SystemExit("lerobot-train is not installed. Run: uv pip install lerobot")
    command = [
        "lerobot-train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={args.dataset_root}",
        f"--policy.type={args.policy}",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.policy}_{args.dataset_repo_id.split('/')[-1]}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        "--trackio.enable=false",
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
