"""Write the simulated station out as a standalone MJCF folder.

    python -m i2rt.simulation.export --out /tmp/yam_station
    python -m mujoco.viewer --mjcf=/tmp/yam_station/scene.xml

The folder holds ``scene.xml`` plus every mesh and texture it refers to, with relative paths, so it
can be moved, opened in MuJoCo's own viewer or loaded from other tools. What an XML file cannot
carry stays behind in this package: the lens and sensor models of the two cameras, the soft shadow
layer and the per-camera paints (``rendering.py``), and the arm's control law (``env.py``).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import mujoco
import tyro

from i2rt.simulation import scene


@dataclass
class Args:
    out: Path
    backdrop: bool = True


def main(args: Args) -> None:
    spec = scene.build_spec(scene.SceneOptions(backdrop=args.backdrop))
    spec.compile()
    xml = spec.to_xml()
    assets = args.out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for kind in (spec.meshes, spec.textures):
        for asset in kind:
            if asset.file:
                source = Path(asset.file)
                shutil.copy2(source, assets / source.name)
                xml = xml.replace(str(source), f"assets/{source.name}")
    (args.out / "scene.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(args.out / "scene.xml"))  # prove it loads on its own
    print(
        f"wrote {args.out / 'scene.xml'} ({model.nbody} bodies, {model.ngeom} geoms, {len(list(assets.iterdir()))} asset files)"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
