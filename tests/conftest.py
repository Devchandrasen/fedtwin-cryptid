from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def vcsl_metadata_dir(tmp_path: Path) -> Path:
    root = tmp_path / "vcsl_metadata"
    root.mkdir()
    splits = {
        "train": [("q1", "r1"), ("q2", "r2")],
        "val": [("q3", "r3")],
        "test": [("q4", "r4"), ("q5", "r5")],
    }
    for split, rows in splits.items():
        pd.DataFrame(rows, columns=["query_id", "reference_id"]).to_csv(root / f"pair_file_{split}.csv", index=False)
    ids = [f"{prefix}{index}" for index in range(1, 6) for prefix in ("q", "r")]
    pd.DataFrame({"uuid": ids, "frame_count": [80 + 3 * index for index in range(len(ids))]}).to_csv(
        root / "frames_all.csv", index=False
    )
    pd.DataFrame({"uuid": ids, "url": [f"https://example.invalid/{value}" for value in ids]}).to_csv(
        root / "videos_url_uuid.csv", index=False
    )
    categories = {"cat_a": ids[::2], "cat_b": ids[1::2]}
    (root / "video_categories.json").write_text(json.dumps(categories), encoding="utf-8")
    return root
