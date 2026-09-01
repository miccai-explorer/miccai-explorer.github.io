"""Save/load/resume checkpoint state for the scraper."""

import json
from pathlib import Path

CHECKPOINT_DIR = Path("data/raw/checkpoints")


def checkpoint_path(year: int) -> Path:
    return CHECKPOINT_DIR / f"miccai_{year}_checkpoint.json"


def load_checkpoint(year: int) -> dict:
    path = checkpoint_path(year)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_ids": [], "papers": []}


def save_checkpoint(year: int, state: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(year)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def clear_checkpoint(year: int) -> None:
    path = checkpoint_path(year)
    if path.exists():
        path.unlink()
