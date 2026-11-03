"""Step 0: build arnio's C++ extension once from BASE_REF and stash it in workspace/ext.

Needs a C++ toolchain (MSVC Build Tools on Windows). Instances only change Python
files, so every clean/buggy checkout reuses this one compiled module.

    python -m pipeline.setup_ext
"""

import shutil
import subprocess
import zipfile

from . import config
from .gitutil import ensure_clone, worktree


def main() -> None:
    ensure_clone()
    src = worktree("build-ext", config.BASE_REF)
    wheels = config.WORKSPACE / "wheels"
    shutil.rmtree(wheels, ignore_errors=True)
    subprocess.run(
        [config.PYTHON, "-m", "pip", "wheel", str(src), "--no-deps", "-w", str(wheels)],
        check=True,
    )
    (wheel,) = wheels.glob("arnio-*.whl")
    config.EXT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            if name.startswith("arnio/_arnio_cpp"):
                (config.EXT_DIR / name.split("/")[-1]).write_bytes(zf.read(name))
                print("extracted", name)


if __name__ == "__main__":
    main()
