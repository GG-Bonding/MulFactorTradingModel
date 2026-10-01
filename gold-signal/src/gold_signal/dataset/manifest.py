"""Write the small dataset manifest the archive loader can read."""

from __future__ import annotations

from pathlib import Path


def write_manifest(path: Path, fields: dict[str, str]) -> None:
    lines = [f"{key}: {value}" for key, value in fields.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
