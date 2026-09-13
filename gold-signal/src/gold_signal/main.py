from __future__ import annotations

import argparse

from gold_signal import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gold Realtime Signal Engine V0")
    parser.add_argument("--mode", choices=("replay", "live"), required=True)
    args = parser.parse_args(argv)
    print(f"Gold Signal Engine V0 {__version__} mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
