from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_branch_inpainting.workflow import train_main


def main(argv=None) -> None:
    train_main(argv)


if __name__ == "__main__":
    main()
