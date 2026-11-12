"""Loads per-target artifacts (columns, categories, metrics, meta)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TARGETS = ("rent", "sale")


@dataclass
class ModelArtifacts:
    target: str
    dir: Path
    columns: list[str]
    meta: dict
    metrics: dict

    @property
    def categories(self) -> dict[str, list[str]]:
        return self.meta["categories"]


def load_artifacts(root: Path) -> dict[str, ModelArtifacts]:
    out = {}
    for t in TARGETS:
        d = root / t
        if not (d / "columns.json").exists():
            continue
        out[t] = ModelArtifacts(
            target=t, dir=d,
            columns=json.loads((d / "columns.json").read_text()),
            meta=json.loads((d / "meta.json").read_text()),
            metrics=json.loads((d / "metrics.json").read_text()),
        )
    if not out:
        raise RuntimeError(f"No model artifacts in {root}. Run `python ml/train.py --target rent|sale`.")
    return out
