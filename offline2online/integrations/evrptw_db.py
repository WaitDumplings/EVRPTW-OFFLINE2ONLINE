from __future__ import annotations

import os
from pathlib import Path
import sys


DEFAULT_EVRPTW_DB_ROOT = Path(
    os.environ.get("EVRPTW_DB_ROOT", "/data/Maojie/EVRPTW-DB")
).resolve()


def configure_evrptw_db(root: str | Path | None = None) -> Path:
    """Make EVRPTW-DB importable without copying its dataset/runtime code."""
    db_root = Path(root).resolve() if root else DEFAULT_EVRPTW_DB_ROOT
    for path in (
        db_root,
        db_root / "EVRPTW_Core",
        db_root / "EVRPTW_Dataset_Generator",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return db_root


def resolve_db_path(path: str | Path | None, db_root: str | Path | None = None) -> Path | None:
    if path is None or str(path) == "":
        return None
    root = Path(db_root).resolve() if db_root else DEFAULT_EVRPTW_DB_ROOT
    out = Path(path)
    return out if out.is_absolute() else root / out
