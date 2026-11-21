"""Install the car desktop launcher after the target checks pass."""

import os
from pathlib import Path
import sys


def main():
    rlinf_root = Path(sys.argv[1]).resolve()
    num_subtasks = int(os.environ.get("TIANJI_CAR_NUM_SUBTASKS", "15"))
    default_banner = (
        "Stage2-30HZ-sub6-1118v1" if num_subtasks == 6 else "Stage2-30HZ-1118v1"
    )
    banner = os.environ.get("TIANJI_CAR_BANNER", default_banner)
    mode_flag = sys.argv[2] if len(sys.argv) > 2 else "--run"
    target = Path.home() / "Desktop" / f"{banner}.desktop"
    entry = str(rlinf_root / "deploy/tianji/stage2_30hz_1118v1.sh")
    if any(char in entry for char in '"\\`$%\r\n'):
        raise ValueError("Unsupported characters in desktop launcher path")
    content = (
        "[Desktop Entry]\nVersion=1.0\nType=Application\n"
        f"Name={banner}\nComment=小车装配 · 固定 Policy · {num_subtasks} 阶段 HIL\n"
        f'Exec=/usr/bin/bash "{entry}" {mode_flag}\n'
        "Terminal=true\nIcon=utilities-terminal\nCategories=Utility;\n"
    )
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        if target.read_text() != content:
            raise FileExistsError(
                f"Review existing desktop entry before replacing {target}"
            )
    else:
        with target.open("x") as handle:
            handle.write(content)
    target.chmod(0o755)
    print(target)


if __name__ == "__main__":
    main()
