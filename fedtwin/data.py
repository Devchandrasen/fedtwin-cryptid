"""Synthetic and feature-level dataset generation for FedTwin-CryptID."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import cosine, l2_normalize
from .pairs import construct_asset_disjoint_pairs
from .transformations import TRANSFORMATIONS, Transformation


@dataclass
class BenchmarkData:
    x_train: np.ndarray
    y_train: np.ndarray
    client_train: np.ndarray
    scenario_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    client_test: np.ndarray
    scenario_test: np.ndarray
    feature_names: list[str]
    manifest: dict


def _one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=float)
    out[index] = 1.0
    return out


def _sample_transformation(rng: np.random.Generator, positive: bool) -> tuple[int, Transformation]:
    if positive:
        pool = list(range(len(TRANSFORMATIONS) - 1))
        weights = np.array([0.18, 0.14, 0.12, 0.12, 0.10, 0.10, 0.10, 0.08, 0.16])
        weights = weights / weights.sum()
        idx = int(rng.choice(pool, p=weights))
    else:
        if rng.random() < 0.30:
            idx = len(TRANSFORMATIONS) - 1
        else:
            idx = int(rng.integers(0, len(TRANSFORMATIONS) - 1))
    return idx, TRANSFORMATIONS[idx]


def _make_pair_features(
    rng: np.random.Generator,
    ref_v: np.ndarray,
    ref_a: np.ndarray,
    query_v: np.ndarray,
    query_a: np.ndarray,
    transform_index: int,
    transform: Transformation,
    positive: bool,
    client_id: int,
    clients: int,
) -> np.ndarray:
    video_score = float(cosine(ref_v[None, :], query_v[None, :])[0])
    audio_score = float(cosine(ref_a[None, :], query_a[None, :])[0])
    temporal_score = 0.78 * video_score + 0.22 * rng.normal(0.0, 0.08)
    metadata_score = (0.72 + rng.normal(0.0, 0.08)) if positive else rng.normal(0.25, 0.20)
    metadata_score = float(np.clip(metadata_score, -1.0, 1.0))
    video_reliability = np.clip(transform.video_reliability + rng.normal(0.0, 0.04), 0.0, 1.0)
    audio_reliability = np.clip(transform.audio_reliability + rng.normal(0.0, 0.04), 0.0, 1.0)
    modality_gap = abs(video_score - audio_score)
    fusion_prior = 0.5 * video_score * video_reliability + 0.5 * audio_score * audio_reliability
    one_hot = _one_hot(transform_index, len(TRANSFORMATIONS))
    client_norm = np.array([client_id / max(clients - 1, 1)], dtype=float)
    return np.concatenate(
        [
            np.array(
                [
                    video_score,
                    audio_score,
                    temporal_score,
                    metadata_score,
                    video_reliability,
                    audio_reliability,
                    modality_gap,
                    fusion_prior,
                ],
                dtype=float,
            ),
            one_hot,
            client_norm,
        ]
    )


def generate_synthetic_benchmark(
    *,
    clients: int = 5,
    assets: int = 1000,
    queries: int = 1000,
    dim: int = 32,
    positive_rate: float = 0.5,
    noniid_alpha: float = 0.3,
    difficulty: float = 1.0,
    test_fraction: float = 0.25,
    seed: int = 7,
) -> BenchmarkData:
    """Generate a deterministic feature-level piracy benchmark.

    The generator simulates reference assets, transformed positive copies, hard
    negatives, modality degradation, and non-IID client assignment. It is not a
    substitute for real media benchmarks, but it exercises the full pipeline.
    """

    rng = np.random.default_rng(seed)
    ref_v = l2_normalize(rng.normal(size=(assets, dim)))
    ref_a = l2_normalize(rng.normal(size=(assets, dim)))

    scenario_bias = rng.dirichlet(np.full(len(TRANSFORMATIONS), noniid_alpha), size=clients)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    client_ids: list[int] = []
    scenarios: list[str] = []

    for _qid in range(queries):
        client_id = int(rng.integers(0, clients))
        positive = bool(rng.random() < positive_rate)
        if positive:
            ref_idx = int(rng.integers(0, assets))
            # Mix global and client-specific transformation tendencies.
            if rng.random() < 0.35:
                t_idx = int(rng.choice(np.arange(len(TRANSFORMATIONS)), p=scenario_bias[client_id]))
                if t_idx == len(TRANSFORMATIONS) - 1:
                    t_idx = int(rng.integers(0, len(TRANSFORMATIONS) - 1))
                transform = TRANSFORMATIONS[t_idx]
            else:
                t_idx, transform = _sample_transformation(rng, positive=True)
            v_noise = rng.normal(0.0, transform.video_strength * difficulty, size=dim)
            a_noise = rng.normal(0.0, transform.audio_strength * difficulty, size=dim)
            query_v = ref_v[ref_idx] + v_noise
            query_a = ref_a[ref_idx] + a_noise
            if transform.name == "audio_replaced":
                query_a = rng.normal(size=dim)
        else:
            ref_idx = int(rng.integers(0, assets))
            other_idx = int((ref_idx + rng.integers(1, assets)) % assets)
            t_idx, transform = _sample_transformation(rng, positive=False)
            if transform.name == "hard_negative_same_topic":
                same_topic_pull = min(0.70, 0.35 + 0.16 * difficulty)
                query_v = same_topic_pull * ref_v[ref_idx] + (1.0 - same_topic_pull) * ref_v[other_idx] + rng.normal(0, 0.38, size=dim)
                query_a = same_topic_pull * ref_a[ref_idx] + (1.0 - same_topic_pull) * ref_a[other_idx] + rng.normal(0, 0.38, size=dim)
            else:
                confusing_pull = max(0.0, min(0.55, 0.10 * difficulty))
                query_v = confusing_pull * ref_v[ref_idx] + (1.0 - confusing_pull) * ref_v[other_idx] + rng.normal(0, 0.72, size=dim)
                query_a = confusing_pull * ref_a[ref_idx] + (1.0 - confusing_pull) * ref_a[other_idx] + rng.normal(0, 0.72, size=dim)

        query_v = l2_normalize(query_v[None, :])[0]
        query_a = l2_normalize(query_a[None, :])[0]
        rows.append(
            _make_pair_features(
                rng,
                ref_v[ref_idx],
                ref_a[ref_idx],
                query_v,
                query_a,
                t_idx,
                transform,
                positive,
                client_id,
                clients,
            )
        )
        labels.append(int(positive))
        client_ids.append(client_id)
        scenarios.append(transform.name)

    x = np.vstack(rows)
    y = np.asarray(labels, dtype=int)
    client_arr = np.asarray(client_ids, dtype=int)
    scenario_arr = np.asarray(scenarios, dtype=object)

    order = rng.permutation(len(y))
    test_size = max(1, int(round(test_fraction * len(y))))
    test_idx = order[:test_size]
    train_idx = order[test_size:]

    feature_names = [
        "video_score",
        "audio_score",
        "temporal_score",
        "metadata_score",
        "video_reliability",
        "audio_reliability",
        "modality_gap",
        "fusion_prior",
    ]
    feature_names += [f"transform_{t.name}" for t in TRANSFORMATIONS]
    feature_names += ["client_norm"]

    manifest = {
        "generator": "synthetic_feature_piracy",
        "clients": clients,
        "assets": assets,
        "queries": queries,
        "embedding_dim": dim,
        "positive_rate": positive_rate,
        "noniid_alpha": noniid_alpha,
        "difficulty": difficulty,
        "test_fraction": test_fraction,
        "seed": seed,
        "transformations": [t.name for t in TRANSFORMATIONS],
    }

    return BenchmarkData(
        x_train=x[train_idx],
        y_train=y[train_idx],
        client_train=client_arr[train_idx],
        scenario_train=scenario_arr[train_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
        client_test=client_arr[test_idx],
        scenario_test=scenario_arr[test_idx],
        feature_names=feature_names,
        manifest=manifest,
    )


def dataframe_from_split(x: np.ndarray, y: np.ndarray, clients: np.ndarray, scenarios: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(x, columns=feature_names)
    df["label"] = y
    df["client_id"] = clients
    df["scenario"] = scenarios
    return df


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _hash_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _hash_vector(text: str, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(_hash_seed(text, seed))
    return l2_normalize(rng.normal(size=(1, dim)))[0]


def _load_vcsl_metadata(metadata_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, int]]:
    base = Path(metadata_dir)
    train = pd.read_csv(base / "pair_file_train.csv")
    val = pd.read_csv(base / "pair_file_val.csv")
    test = pd.read_csv(base / "pair_file_test.csv")
    frames = pd.read_csv(base / "frames_all.csv")
    with open(base / "video_categories.json", encoding="utf-8") as fh:
        categories = json.load(fh)
    category_by_uuid: dict[str, str] = {}
    for category, ids in categories.items():
        for uuid in ids:
            category_by_uuid[str(uuid)] = str(category)
    frame_count = dict(zip(frames["uuid"].astype(str), frames["frame_count"].astype(int), strict=True))
    return train, val, test, category_by_uuid, frame_count


def _build_vcsl_groups(*pair_frames: pd.DataFrame) -> _UnionFind:
    uf = _UnionFind()
    for frame in pair_frames:
        for q, r in zip(frame["query_id"].astype(str), frame["reference_id"].astype(str), strict=True):
            uf.union(q, r)
    return uf


def _clean_vcsl_positive_pairs(*pair_frames: pd.DataFrame) -> pd.DataFrame:
    """Normalize VCSL positives and remove trivial identity pairs."""

    frame = pd.concat(pair_frames, ignore_index=True)[["query_id", "reference_id"]].copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["reference_id"] = frame["reference_id"].astype(str)
    frame = frame[frame["query_id"].ne(frame["reference_id"])]
    return frame.drop_duplicates(["query_id", "reference_id"], keep="first").reset_index(drop=True)


def _sample_vcsl_pairs(
    positives: pd.DataFrame,
    all_ids: list[str],
    category_by_uuid: dict[str, str],
    uf: _UnionFind,
    rng: np.random.Generator,
    *,
    max_positive_pairs: int,
    negative_ratio: float,
    min_positive_queries: int = 0,
) -> pd.DataFrame:
    pos = positives.copy()
    if min_positive_queries < 0:
        raise ValueError("min_positive_queries must be non-negative")
    available_queries = pos["query_id"].astype(str).nunique()
    if min_positive_queries > available_queries:
        raise ValueError(
            f"requested {min_positive_queries} positive-bearing queries, "
            f"but only {available_queries} are available"
        )
    if max_positive_pairs and max_positive_pairs < min_positive_queries:
        raise ValueError("max_positive_pairs cannot be smaller than min_positive_queries")
    if max_positive_pairs and len(pos) > max_positive_pairs:
        if min_positive_queries:
            query_values = pos["query_id"].astype(str)
            selected_queries = rng.choice(
                np.asarray(sorted(query_values.unique())),
                size=min_positive_queries,
                replace=False,
            )
            anchor_indices = []
            for query_id in selected_queries:
                candidates = pos.index[query_values.eq(str(query_id))].to_numpy()
                anchor_indices.append(int(rng.choice(candidates)))
            anchors = pos.loc[anchor_indices]
            remaining = pos.drop(index=anchor_indices)
            fill_count = max_positive_pairs - len(anchors)
            if fill_count:
                fill = remaining.sample(
                    n=fill_count,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
                pos = pd.concat([anchors, fill], ignore_index=True)
            else:
                pos = anchors.reset_index(drop=True)
            pos = pos.sample(
                frac=1.0,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).reset_index(drop=True)
        else:
            pos = pos.sample(n=max_positive_pairs, random_state=int(rng.integers(0, 2**31 - 1)))
    if pos["query_id"].astype(str).nunique() < min_positive_queries:
        raise RuntimeError("positive-pair sampling failed the distinct-query invariant")
    pos = pos.assign(label=1)
    id_by_category: dict[str, list[str]] = {}
    for vid in all_ids:
        id_by_category.setdefault(category_by_uuid.get(vid, "unknown"), []).append(vid)

    neg_rows = []
    neg_count = int(round(len(pos) * negative_ratio))
    pos_queries = pos["query_id"].astype(str).to_numpy()
    for _ in range(neg_count):
        q = str(rng.choice(pos_queries))
        q_cat = category_by_uuid.get(q, "unknown")
        hard = rng.random() < 0.85
        pool = id_by_category.get(q_cat, all_ids) if hard else all_ids
        for _attempt in range(32):
            r = str(rng.choice(pool))
            if r != q and uf.find(r) != uf.find(q):
                break
        else:
            r = str(rng.choice(all_ids))
        neg_rows.append({"query_id": q, "reference_id": r, "label": 0})
    neg = pd.DataFrame(neg_rows)
    return pd.concat([pos[["query_id", "reference_id", "label"]], neg], ignore_index=True)


def _vcsl_pair_to_features(
    pair_df: pd.DataFrame,
    category_by_uuid: dict[str, str],
    frame_count: dict[str, int],
    uf: _UnionFind,
    *,
    clients: int,
    dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    category_names = sorted(set(category_by_uuid.values()) | {"unknown"})
    cat_index = {name: i for i, name in enumerate(category_names)}
    rows = []
    labels = []
    client_ids = []
    scenarios = []
    rng = np.random.default_rng(seed)
    vector_cache: dict[str, np.ndarray] = {}

    def cached_hash_vector(key: str) -> np.ndarray:
        if key not in vector_cache:
            vector_cache[key] = _hash_vector(key, dim, seed)
        return vector_cache[key]

    max_frame = max(frame_count.values()) if frame_count else 1
    for q, r, label in zip(
        pair_df["query_id"].astype(str),
        pair_df["reference_id"].astype(str),
        pair_df["label"].astype(int),
        strict=True,
    ):
        q_group = uf.find(q)
        r_group = uf.find(r)
        q_cat = category_by_uuid.get(q, "unknown")
        r_cat = category_by_uuid.get(r, "unknown")
        q_frame = frame_count.get(q, int(max_frame * 0.25))
        r_frame = frame_count.get(r, int(max_frame * 0.25))
        q_cat_v = cached_hash_vector(f"video-category:{q_cat}")
        r_cat_v = cached_hash_vector(f"video-category:{r_cat}")
        q_base_v = cached_hash_vector(f"video-group:{q_group}")
        r_base_v = cached_hash_vector(f"video-group:{r_group}")
        q_vid = l2_normalize(
            (
                0.34 * q_cat_v
                + 0.52 * q_base_v
                + 0.34 * cached_hash_vector(f"video-id:{q}")
                + rng.normal(0, 0.10, size=dim)
            )[None, :]
        )[0]
        r_vid = l2_normalize(
            (
                0.34 * r_cat_v
                + 0.52 * r_base_v
                + 0.34 * cached_hash_vector(f"video-id:{r}")
                + rng.normal(0, 0.10, size=dim)
            )[None, :]
        )[0]

        # VCSL is video-centric; audio here is a deterministic proxy used to
        # stress multimodal fusion and missing/replaced audio behavior.
        q_audio_base = cached_hash_vector(f"audio-cat:{q_cat}")
        r_audio_base = cached_hash_vector(f"audio-cat:{r_cat}")
        q_audio_group = cached_hash_vector(f"audio-group:{q_group}")
        if label:
            q_aud = l2_normalize((0.30 * q_audio_base + 0.42 * q_audio_group + 0.50 * cached_hash_vector(f"audio-id:{q}") + rng.normal(0, 0.13, size=dim))[None, :])[0]
            r_aud = l2_normalize((0.30 * q_audio_base + 0.42 * q_audio_group + 0.50 * cached_hash_vector(f"audio-id:{r}") + rng.normal(0, 0.13, size=dim))[None, :])[0]
        else:
            q_aud = l2_normalize((0.40 * q_audio_base + 0.60 * cached_hash_vector(f"audio-id:{q}") + rng.normal(0, 0.16, size=dim))[None, :])[0]
            r_aud = l2_normalize((0.40 * r_audio_base + 0.60 * cached_hash_vector(f"audio-id:{r}") + rng.normal(0, 0.16, size=dim))[None, :])[0]

        video_score = float(cosine(q_vid[None, :], r_vid[None, :])[0])
        audio_score = float(cosine(q_aud[None, :], r_aud[None, :])[0])
        frame_ratio = min(q_frame, r_frame) / max(max(q_frame, r_frame), 1)
        temporal_score = 0.72 * video_score + 0.28 * frame_ratio
        same_category = float(q_cat == r_cat)
        metadata_score = 0.7 * same_category + 0.3 * frame_ratio
        video_reliability = float(np.clip(0.72 + 0.25 * frame_ratio + rng.normal(0, 0.03), 0, 1))
        audio_reliability = float(np.clip(0.55 + 0.30 * same_category + rng.normal(0, 0.05), 0, 1))
        modality_gap = abs(video_score - audio_score)
        fusion_prior = 0.55 * video_score * video_reliability + 0.45 * audio_score * audio_reliability
        video_rel_score = video_score * video_reliability
        audio_rel_score = audio_score * audio_reliability
        temporal_rel_score = temporal_score * video_reliability
        max_modality_score = max(video_score, audio_score)
        mean_modality_score = 0.5 * (video_score + audio_score)
        product_modality_score = video_score * audio_score
        category_audio_score = same_category * audio_score
        frame_video_score = frame_ratio * video_score
        cat_onehot = np.zeros(len(category_names), dtype=float)
        cat_onehot[cat_index[q_cat]] = 1.0
        client_id = int(cat_index[q_cat] % max(clients, 1))
        row = np.concatenate(
            [
                np.array(
                    [
                        video_score,
                        audio_score,
                        temporal_score,
                        metadata_score,
                        video_reliability,
                        audio_reliability,
                        modality_gap,
                        fusion_prior,
                        frame_ratio,
                        same_category,
                        video_rel_score,
                        audio_rel_score,
                        temporal_rel_score,
                        max_modality_score,
                        mean_modality_score,
                        product_modality_score,
                        category_audio_score,
                        frame_video_score,
                    ],
                    dtype=float,
                ),
                cat_onehot,
                np.array([client_id / max(clients - 1, 1)], dtype=float),
            ]
        )
        rows.append(row)
        labels.append(label)
        client_ids.append(client_id)
        if label and frame_ratio < 0.45:
            scenarios.append("vcsl_partial_copy")
        elif label:
            scenarios.append("vcsl_labeled_copy")
        elif same_category:
            scenarios.append("vcsl_same_category_negative")
        else:
            scenarios.append("vcsl_cross_category_negative")

    feature_names = [
        "video_score",
        "audio_score",
        "temporal_score",
        "metadata_score",
        "video_reliability",
        "audio_reliability",
        "modality_gap",
        "fusion_prior",
        "frame_ratio",
        "same_category",
        "video_rel_score",
        "audio_rel_score",
        "temporal_rel_score",
        "max_modality_score",
        "mean_modality_score",
        "product_modality_score",
        "category_audio_score",
        "frame_video_score",
    ]
    feature_names += [f"category_{c}" for c in category_names]
    feature_names += ["client_norm"]
    return (
        np.vstack(rows).astype(float),
        np.asarray(labels, dtype=int),
        np.asarray(client_ids, dtype=int),
        np.asarray(scenarios, dtype=object),
        feature_names,
    )


def generate_vcsl_public_benchmark(
    *,
    metadata_dir: str | Path,
    clients: int = 10,
    dim: int = 64,
    max_train_pairs: int = 60000,
    max_test_pairs: int = 30000,
    negative_ratio: float = 1.0,
    seed: int = 31,
) -> BenchmarkData:
    """Build a benchmark from public VCSL labels and metadata.

    This tier uses real VCSL positive pair labels, video IDs, frame counts, and
    categories. The feature vectors are deterministic label-topology features
    when the large VCSL visual feature archives have not yet been staged.
    """

    train, val, test, category_by_uuid, frame_count = _load_vcsl_metadata(metadata_dir)
    all_positive_pairs = _clean_vcsl_positive_pairs(train, val, test)
    uf = _build_vcsl_groups(all_positive_pairs)
    all_ids = sorted(set(frame_count) | set(category_by_uuid))
    split_pairs = construct_asset_disjoint_pairs(
        all_positive_pairs,
        asset_ids=all_ids,
        negative_ratio=0.0,
        test_fraction=0.2,
        seed=seed,
    )
    train_assets = sorted(set(split_pairs.train["query_id"]) | set(split_pairs.train["reference_id"]))
    test_assets = sorted(set(split_pairs.test["query_id"]) | set(split_pairs.test["reference_id"]))
    rng = np.random.default_rng(seed)
    train_pairs = _sample_vcsl_pairs(
        split_pairs.train,
        train_assets,
        category_by_uuid,
        uf,
        rng,
        max_positive_pairs=max_train_pairs,
        negative_ratio=negative_ratio,
    )
    test_pairs = _sample_vcsl_pairs(
        split_pairs.test,
        test_assets,
        category_by_uuid,
        uf,
        rng,
        max_positive_pairs=max_test_pairs,
        negative_ratio=negative_ratio,
    )
    x_train, y_train, client_train, scenario_train, feature_names = _vcsl_pair_to_features(
        train_pairs, category_by_uuid, frame_count, uf, clients=clients, dim=dim, seed=seed
    )
    x_test, y_test, client_test, scenario_test, _ = _vcsl_pair_to_features(
        test_pairs, category_by_uuid, frame_count, uf, clients=clients, dim=dim, seed=seed
    )
    manifest = {
        "generator": "vcsl_public_label_topology",
        "metadata_dir": str(metadata_dir),
        "clients": clients,
        "embedding_dim": dim,
        "max_train_positive_pairs": max_train_pairs,
        "max_test_positive_pairs": max_test_pairs,
        "negative_ratio": negative_ratio,
        "seed": seed,
        "source": "VCSL public GitHub metadata and labels",
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "feature_note": "deterministic public-label topology features; large visual feature archive not required",
        "split_policy": split_pairs.manifest["split_policy"],
        "asset_overlap": split_pairs.manifest["asset_overlap"],
        "train_assets": len(train_assets),
        "test_assets": len(test_assets),
    }
    return BenchmarkData(
        x_train=x_train,
        y_train=y_train,
        client_train=client_train,
        scenario_train=scenario_train,
        x_test=x_test,
        y_test=y_test,
        client_test=client_test,
        scenario_test=scenario_test,
        feature_names=feature_names,
        manifest=manifest,
    )


def _resolve_isc_feature_root(feature_dir: str | Path) -> Path:
    base = Path(feature_dir)
    if (base / "eff256d").is_dir():
        return base / "eff256d"
    return base


def _load_isc_frames(
    video_id: str,
    feature_root: Path,
    cache: dict[str, np.ndarray],
    *,
    max_frames: int,
) -> np.ndarray:
    if video_id in cache:
        return cache[video_id]
    path = feature_root / f"{video_id}.npy"
    arr = np.load(path).astype("float32")
    if len(arr) > max_frames:
        indices = np.linspace(0, len(arr) - 1, max_frames).astype(int)
        arr = arr[indices]
    arr = l2_normalize(arr)
    cache[video_id] = arr
    return arr


def _frame_match_scores(q_frames: np.ndarray, r_frames: np.ndarray) -> tuple[float, float, float, float]:
    sim = q_frames @ r_frames.T
    flat = sim.ravel()
    k = min(10, flat.size)
    topk = np.partition(flat, -k)[-k:]
    mean_score = float(sim.mean())
    max_score = float(sim.max())
    topk_score = float(topk.mean())
    mutual_score = float(0.5 * (sim.max(axis=0).mean() + sim.max(axis=1).mean()))
    return mean_score, max_score, topk_score, mutual_score


def _vcsl_isc_pair_to_features(
    pair_df: pd.DataFrame,
    category_by_uuid: dict[str, str],
    frame_count: dict[str, int],
    uf: _UnionFind,
    *,
    feature_dir: str | Path,
    clients: int,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    category_names = sorted(set(category_by_uuid.values()) | {"unknown"})
    cat_index = {name: i for i, name in enumerate(category_names)}
    feature_root = _resolve_isc_feature_root(feature_dir)
    cache: dict[str, np.ndarray] = {}
    rows = []
    labels = []
    client_ids = []
    scenarios = []

    max_frame = max(frame_count.values()) if frame_count else 1
    for q, r, label in zip(
        pair_df["query_id"].astype(str),
        pair_df["reference_id"].astype(str),
        pair_df["label"].astype(int),
        strict=True,
    ):
        q_cat = category_by_uuid.get(q, "unknown")
        r_cat = category_by_uuid.get(r, "unknown")
        q_frame = frame_count.get(q, int(max_frame * 0.25))
        r_frame = frame_count.get(r, int(max_frame * 0.25))
        q_arr = _load_isc_frames(q, feature_root, cache, max_frames=max_frames)
        r_arr = _load_isc_frames(r, feature_root, cache, max_frames=max_frames)
        mean_score, max_score, topk_score, mutual_score = _frame_match_scores(q_arr, r_arr)

        frame_ratio = min(q_frame, r_frame) / max(max(q_frame, r_frame), 1)
        same_category = float(q_cat == r_cat)
        video_score = 0.70 * topk_score + 0.30 * max_score
        temporal_score = mutual_score
        metadata_score = 0.7 * same_category + 0.3 * frame_ratio
        video_reliability = float(np.clip(0.55 + 0.35 * frame_ratio + 0.10 * min(len(q_arr), len(r_arr)) / max(max_frames, 1), 0, 1))
        # VCSL ISC is visual-only. Audio fields are retained as low-confidence
        # auxiliary slots so the common experiment schema stays intact.
        audio_score = mean_score
        audio_reliability = 0.10
        modality_gap = abs(video_score - audio_score)
        fusion_prior = 0.85 * video_score * video_reliability + 0.15 * audio_score * audio_reliability
        video_rel_score = video_score * video_reliability
        audio_rel_score = audio_score * audio_reliability
        temporal_rel_score = temporal_score * video_reliability
        max_modality_score = max(video_score, audio_score)
        mean_modality_score = 0.5 * (video_score + audio_score)
        product_modality_score = video_score * max(audio_score, 0.0)
        category_audio_score = same_category * audio_score
        frame_video_score = frame_ratio * video_score
        cat_onehot = np.zeros(len(category_names), dtype=float)
        cat_onehot[cat_index[q_cat]] = 1.0
        client_id = int(cat_index[q_cat] % max(clients, 1))
        row = np.concatenate(
            [
                np.array(
                    [
                        video_score,
                        audio_score,
                        temporal_score,
                        metadata_score,
                        video_reliability,
                        audio_reliability,
                        modality_gap,
                        fusion_prior,
                        frame_ratio,
                        same_category,
                        video_rel_score,
                        audio_rel_score,
                        temporal_rel_score,
                        max_modality_score,
                        mean_modality_score,
                        product_modality_score,
                        category_audio_score,
                        frame_video_score,
                    ],
                    dtype=float,
                ),
                cat_onehot,
                np.array([client_id / max(clients - 1, 1)], dtype=float),
            ]
        )
        rows.append(row)
        labels.append(label)
        client_ids.append(client_id)
        if label and frame_ratio < 0.45:
            scenarios.append("isc_partial_copy")
        elif label:
            scenarios.append("isc_labeled_copy")
        elif same_category:
            scenarios.append("isc_same_category_negative")
        else:
            scenarios.append("isc_cross_category_negative")

    feature_names = [
        "video_score",
        "audio_score",
        "temporal_score",
        "metadata_score",
        "video_reliability",
        "audio_reliability",
        "modality_gap",
        "fusion_prior",
        "frame_ratio",
        "same_category",
        "video_rel_score",
        "audio_rel_score",
        "temporal_rel_score",
        "max_modality_score",
        "mean_modality_score",
        "product_modality_score",
        "category_audio_score",
        "frame_video_score",
    ]
    feature_names += [f"category_{c}" for c in category_names]
    feature_names += ["client_norm"]
    return (
        np.vstack(rows).astype(float),
        np.asarray(labels, dtype=int),
        np.asarray(client_ids, dtype=int),
        np.asarray(scenarios, dtype=object),
        feature_names,
    )


def generate_vcsl_isc_benchmark(
    *,
    metadata_dir: str | Path,
    feature_dir: str | Path,
    clients: int = 10,
    max_train_pairs: int = 6000,
    max_test_pairs: int = 3000,
    negative_ratio: float = 1.0,
    max_frames: int = 160,
    seed: int = 31,
) -> BenchmarkData:
    """Build a VCSL benchmark from released ISC frame descriptors."""

    train, val, test, category_by_uuid, frame_count = _load_vcsl_metadata(metadata_dir)
    feature_root = _resolve_isc_feature_root(feature_dir)
    available_ids = {path.stem for path in feature_root.glob("*.npy")}
    all_ids = sorted((set(frame_count) | set(category_by_uuid)) & available_ids)
    all_positive_pairs = _clean_vcsl_positive_pairs(train, val, test)
    all_positive_pairs = all_positive_pairs[
        all_positive_pairs["query_id"].isin(available_ids) & all_positive_pairs["reference_id"].isin(available_ids)
    ].reset_index(drop=True)
    uf = _build_vcsl_groups(all_positive_pairs)
    split_pairs = construct_asset_disjoint_pairs(
        all_positive_pairs,
        asset_ids=all_ids,
        negative_ratio=0.0,
        test_fraction=0.2,
        seed=seed,
    )
    train_assets = sorted(set(split_pairs.train["query_id"]) | set(split_pairs.train["reference_id"]))
    test_assets = sorted(set(split_pairs.test["query_id"]) | set(split_pairs.test["reference_id"]))
    rng = np.random.default_rng(seed)
    train_pairs = _sample_vcsl_pairs(
        split_pairs.train,
        train_assets,
        category_by_uuid,
        uf,
        rng,
        max_positive_pairs=max_train_pairs,
        negative_ratio=negative_ratio,
    )
    test_pairs = _sample_vcsl_pairs(
        split_pairs.test,
        test_assets,
        category_by_uuid,
        uf,
        rng,
        max_positive_pairs=max_test_pairs,
        negative_ratio=negative_ratio,
    )
    x_train, y_train, client_train, scenario_train, feature_names = _vcsl_isc_pair_to_features(
        train_pairs,
        category_by_uuid,
        frame_count,
        uf,
        feature_dir=feature_dir,
        clients=clients,
        max_frames=max_frames,
    )
    x_test, y_test, client_test, scenario_test, _ = _vcsl_isc_pair_to_features(
        test_pairs,
        category_by_uuid,
        frame_count,
        uf,
        feature_dir=feature_dir,
        clients=clients,
        max_frames=max_frames,
    )
    manifest = {
        "generator": "vcsl_released_isc_frame_features",
        "metadata_dir": str(metadata_dir),
        "feature_dir": str(feature_dir),
        "available_feature_files": int(len(available_ids)),
        "clients": clients,
        "max_train_positive_pairs": max_train_pairs,
        "max_test_positive_pairs": max_test_pairs,
        "negative_ratio": negative_ratio,
        "max_frames_per_video": max_frames,
        "seed": seed,
        "source": "VCSL public GitHub metadata plus released ISC frame descriptors",
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "feature_note": "released visual ISC frame descriptors; audio fields are low-confidence schema placeholders",
        "split_policy": split_pairs.manifest["split_policy"],
        "asset_overlap": split_pairs.manifest["asset_overlap"],
        "train_assets": len(train_assets),
        "test_assets": len(test_assets),
    }
    return BenchmarkData(
        x_train=x_train,
        y_train=y_train,
        client_train=client_train,
        scenario_train=scenario_train,
        x_test=x_test,
        y_test=y_test,
        client_test=client_test,
        scenario_test=scenario_test,
        feature_names=feature_names,
        manifest=manifest,
    )


FMA_AUDIO_TRANSFORMS = ["crop", "noise", "lowpass", "dropout", "speed"]


def _load_fma_tracks(metadata_dir: str | Path) -> dict[int, str]:
    base = Path(metadata_dir)
    candidates = [base / "tracks.csv", base / "fma_metadata" / "tracks.csv"]
    tracks_path = next((p for p in candidates if p.exists()), None)
    if tracks_path is None:
        return {}
    tracks = pd.read_csv(tracks_path, header=[0, 1], index_col=0)
    genre_col = ("track", "genre_top")
    if genre_col not in tracks.columns:
        return {}
    genres = tracks[genre_col].fillna("unknown").astype(str)
    return {int(track_id): genre for track_id, genre in genres.items()}


def _discover_fma_audio_files(audio_dir: str | Path) -> dict[int, Path]:
    base = Path(audio_dir)
    files = {}
    for path in base.rglob("*.mp3"):
        try:
            track_id = int(path.stem)
        except ValueError:
            continue
        files[track_id] = path
    return files


def _decode_mp3(path: Path, *, sample_rate: int, max_seconds: float) -> np.ndarray:
    import miniaudio

    decoded = miniaudio.decode_file(
        str(path),
        output_format=miniaudio.SampleFormat.FLOAT32,
        nchannels=1,
        sample_rate=sample_rate,
    )
    y = np.asarray(decoded.samples, dtype=np.float32)
    max_samples = int(sample_rate * max_seconds)
    if max_samples > 0:
        y = y[:max_samples]
    if len(y) < sample_rate:
        y = np.pad(y, (0, max(0, sample_rate - len(y))))
    y = y.astype(np.float32)
    y = y - float(np.mean(y))
    scale = float(np.max(np.abs(y)))
    if scale > 1e-8:
        y = y / scale
    return y


def _audio_transform(y: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    out = y.astype(np.float32).copy()
    if name == "crop":
        keep = max(1024, int(0.70 * len(out)))
        start = int(rng.integers(0, max(1, len(out) - keep + 1)))
        out = out[start : start + keep]
    elif name == "noise":
        power = float(np.mean(out**2)) + 1e-8
        noise = rng.normal(0.0, np.sqrt(power / (10 ** (18 / 10))), size=len(out)).astype(np.float32)
        out = out + noise
    elif name == "lowpass":
        kernel = np.ones(9, dtype=np.float32) / 9.0
        out = np.convolve(out, kernel, mode="same").astype(np.float32)
    elif name == "dropout":
        span = max(512, int(0.18 * len(out)))
        start = int(rng.integers(0, max(1, len(out) - span + 1)))
        out[start : start + span] = 0.0
    elif name == "speed":
        factor = float(rng.choice([0.94, 1.06]))
        src = np.arange(len(out), dtype=float)
        target = np.arange(0, len(out), factor, dtype=float)
        out = np.interp(target, src, out).astype(np.float32)
    out = out - float(np.mean(out))
    scale = float(np.max(np.abs(out)))
    if scale > 1e-8:
        out = out / scale
    return out


def _frame_audio(y: np.ndarray, frame_len: int = 512, hop: int = 256) -> np.ndarray:
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    n = 1 + (len(y) - frame_len) // hop
    starts = np.arange(n)[:, None] * hop
    offsets = np.arange(frame_len)[None, :]
    frames = y[starts + offsets]
    return frames * np.hanning(frame_len)[None, :]


def _audio_descriptor(y: np.ndarray, sample_rate: int) -> np.ndarray:
    frames = _frame_audio(y)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    spec += 1e-10
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)

    edges = np.geomspace(60, sample_rate / 2, 33)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        idx = (freqs >= lo) & (freqs < hi)
        if not idx.any():
            bands.append(np.zeros(spec.shape[0], dtype=float))
        else:
            bands.append(np.log1p(spec[:, idx].mean(axis=1)))
    band_arr = np.vstack(bands).T
    band_mean = band_arr.mean(axis=0)
    band_std = band_arr.std(axis=0)

    valid = freqs > 40
    midi = np.round(69 + 12 * np.log2(np.maximum(freqs[valid], 1e-6) / 440.0)).astype(int)
    pitch_class = np.mod(midi, 12)
    chroma = np.zeros((spec.shape[0], 12), dtype=float)
    for pc in range(12):
        idx = np.zeros_like(freqs, dtype=bool)
        idx[valid] = pitch_class == pc
        if idx.any():
            chroma[:, pc] = spec[:, idx].sum(axis=1)
    chroma = chroma / np.maximum(chroma.sum(axis=1, keepdims=True), 1e-10)

    energy = spec.sum(axis=1)
    centroid = (spec * freqs[None, :]).sum(axis=1) / np.maximum(energy, 1e-10)
    centroid_norm = centroid / max(sample_rate / 2, 1)
    bandwidth = np.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * spec).sum(axis=1) / np.maximum(energy, 1e-10))
    bandwidth_norm = bandwidth / max(sample_rate / 2, 1)
    cumulative = np.cumsum(spec, axis=1)
    roll_idx = np.argmax(cumulative >= 0.85 * cumulative[:, -1:], axis=1)
    rolloff = freqs[roll_idx] / max(sample_rate / 2, 1)
    zcr = (np.abs(np.diff(np.signbit(frames), axis=1)).mean(axis=1)).astype(float)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    flux = np.sqrt(np.mean(np.diff(np.log1p(spec), axis=0) ** 2, axis=1)) if spec.shape[0] > 1 else np.zeros(1)

    stats = np.array(
        [
            centroid_norm.mean(),
            centroid_norm.std(),
            bandwidth_norm.mean(),
            bandwidth_norm.std(),
            rolloff.mean(),
            rolloff.std(),
            zcr.mean(),
            zcr.std(),
            rms.mean(),
            rms.std(),
            flux.mean(),
            flux.std(),
        ],
        dtype=float,
    )
    desc = np.concatenate([band_mean, band_std, chroma.mean(axis=0), chroma.std(axis=0), stats])
    return l2_normalize(desc[None, :])[0].astype(np.float32)


def _prepare_fma_audio_cache(
    *,
    audio_dir: str | Path,
    metadata_dir: str | Path,
    cache_dir: str | Path,
    max_tracks: int,
    sample_rate: int,
    max_seconds: float,
    seed: int,
) -> dict:
    cache_base = Path(cache_dir)
    cache_base.mkdir(parents=True, exist_ok=True)
    cache_path = cache_base / f"fma_audio_sr{sample_rate}_sec{int(max_seconds)}_tracks{max_tracks}.npz"
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        return {
            "track_ids": loaded["track_ids"],
            "genres": loaded["genres"],
            "clients": loaded["clients"],
            "descriptors": loaded["descriptors"],
            "transforms": list(loaded["transforms"]),
            "cache_path": str(cache_path),
        }

    genres_by_id = _load_fma_tracks(metadata_dir)
    audio_files = _discover_fma_audio_files(audio_dir)
    common_ids = sorted(set(audio_files) & set(genres_by_id))
    if not common_ids:
        raise FileNotFoundError(f"No FMA mp3 files matched metadata under {audio_dir}")

    by_genre: dict[str, list[int]] = {}
    for track_id in common_ids:
        by_genre.setdefault(genres_by_id.get(track_id, "unknown"), []).append(track_id)
    selected: list[int] = []
    genre_names = sorted(by_genre)
    while len(selected) < min(max_tracks, len(common_ids)):
        progressed = False
        for genre in genre_names:
            bucket = by_genre[genre]
            if bucket:
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= min(max_tracks, len(common_ids)):
                    break
        if not progressed:
            break
    selected = sorted(selected)
    genre_index = {genre: i for i, genre in enumerate(genre_names)}
    transform_names = ["clean"] + FMA_AUDIO_TRANSFORMS
    descriptors = []
    kept_ids = []
    kept_genres = []
    kept_clients = []
    for track_id in selected:
        try:
            y = _decode_mp3(audio_files[track_id], sample_rate=sample_rate, max_seconds=max_seconds)
            track_desc = [_audio_descriptor(y, sample_rate)]
            for transform_name in FMA_AUDIO_TRANSFORMS:
                trng = np.random.default_rng(_hash_seed(f"fma:{track_id}:{transform_name}", 0))
                track_desc.append(_audio_descriptor(_audio_transform(y, transform_name, trng), sample_rate))
            descriptors.append(np.vstack(track_desc))
            genre = genres_by_id.get(track_id, "unknown")
            kept_ids.append(track_id)
            kept_genres.append(genre)
            kept_clients.append(int(genre_index.get(genre, 0) % 32))
        except Exception:
            continue
    if not descriptors:
        raise RuntimeError("FMA descriptor cache could not decode any tracks")

    out = {
        "track_ids": np.asarray(kept_ids, dtype=int),
        "genres": np.asarray(kept_genres, dtype=object),
        "clients": np.asarray(kept_clients, dtype=int),
        "descriptors": np.asarray(descriptors, dtype=np.float32),
        "transforms": np.asarray(transform_names, dtype=object),
        "cache_path": str(cache_path),
    }
    np.savez_compressed(cache_path, **out)
    return {**out, "transforms": list(transform_names)}


def _sample_fma_pairs(
    indices: np.ndarray,
    genres: np.ndarray,
    rng: np.random.Generator,
    *,
    max_positive_pairs: int,
    negative_ratio: float,
) -> list[tuple[int, int, int, int]]:
    by_genre: dict[str, list[int]] = {}
    for idx in indices:
        by_genre.setdefault(str(genres[idx]), []).append(int(idx))
    positives = []
    for _ in range(max_positive_pairs):
        idx = int(rng.choice(indices))
        transform_idx = int(rng.integers(1, len(FMA_AUDIO_TRANSFORMS) + 1))
        positives.append((idx, idx, transform_idx, 1))
    negatives = []
    neg_count = int(round(max_positive_pairs * negative_ratio))
    for _ in range(neg_count):
        q = int(rng.choice(indices))
        hard = rng.random() < 0.90
        pool = by_genre.get(str(genres[q]), list(map(int, indices))) if hard else list(map(int, indices))
        for _attempt in range(64):
            r = int(rng.choice(pool))
            if r != q:
                break
        else:
            r = int(rng.choice(indices))
            if r == q:
                r = int(indices[(np.where(indices == q)[0][0] + 1) % len(indices)])
        negatives.append((q, r, 0, 0))
    return positives + negatives


def _fma_pairs_to_features(
    pairs: list[tuple[int, int, int, int]],
    descriptors: np.ndarray,
    genres: np.ndarray,
    clients: np.ndarray,
    *,
    client_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    genre_names = sorted(set(map(str, genres)) | {"unknown"})
    genre_index = {genre: i for i, genre in enumerate(genre_names)}
    rows = []
    labels = []
    client_ids = []
    scenarios = []
    reliability = {"crop": 0.78, "noise": 0.82, "lowpass": 0.86, "dropout": 0.70, "speed": 0.76}
    transform_names = ["clean"] + FMA_AUDIO_TRANSFORMS
    for q, r, transform_idx, label in pairs:
        q_desc = descriptors[q, 0]
        r_desc = descriptors[r, transform_idx]
        audio_score = float(cosine(q_desc[None, :], r_desc[None, :])[0])
        same_genre = float(str(genres[q]) == str(genres[r]))
        transform_name = transform_names[transform_idx]
        audio_reliability = reliability.get(transform_name, 0.55) if label else 0.72
        video_score = 0.05
        video_reliability = 0.05
        temporal_score = audio_score
        metadata_score = 0.7 * same_genre
        modality_gap = abs(audio_score - video_score)
        fusion_prior = 0.92 * audio_score * audio_reliability + 0.08 * video_score * video_reliability
        video_rel_score = video_score * video_reliability
        audio_rel_score = audio_score * audio_reliability
        temporal_rel_score = temporal_score * audio_reliability
        max_modality_score = max(video_score, audio_score)
        mean_modality_score = 0.5 * (video_score + audio_score)
        product_modality_score = video_score * audio_score
        category_audio_score = same_genre * audio_score
        frame_video_score = video_score
        genre_onehot = np.zeros(len(genre_names), dtype=float)
        genre_onehot[genre_index.get(str(genres[q]), genre_index["unknown"])] = 1.0
        client_id = int(clients[q] % max(client_count, 1))
        row = np.concatenate(
            [
                np.array(
                    [
                        video_score,
                        audio_score,
                        temporal_score,
                        metadata_score,
                        video_reliability,
                        audio_reliability,
                        modality_gap,
                        fusion_prior,
                        1.0,
                        same_genre,
                        video_rel_score,
                        audio_rel_score,
                        temporal_rel_score,
                        max_modality_score,
                        mean_modality_score,
                        product_modality_score,
                        category_audio_score,
                        frame_video_score,
                    ],
                    dtype=float,
                ),
                genre_onehot,
                np.array([client_id / max(client_count - 1, 1)], dtype=float),
            ]
        )
        rows.append(row)
        labels.append(label)
        client_ids.append(client_id)
        if label:
            scenarios.append(f"fma_{transform_name}_copy")
        elif same_genre:
            scenarios.append("fma_same_genre_negative")
        else:
            scenarios.append("fma_cross_genre_negative")

    feature_names = [
        "video_score",
        "audio_score",
        "temporal_score",
        "metadata_score",
        "video_reliability",
        "audio_reliability",
        "modality_gap",
        "fusion_prior",
        "frame_ratio",
        "same_category",
        "video_rel_score",
        "audio_rel_score",
        "temporal_rel_score",
        "max_modality_score",
        "mean_modality_score",
        "product_modality_score",
        "category_audio_score",
        "frame_video_score",
    ]
    feature_names += [f"category_{name}" for name in genre_names]
    feature_names += ["client_norm"]
    return (
        np.vstack(rows).astype(float),
        np.asarray(labels, dtype=int),
        np.asarray(client_ids, dtype=int),
        np.asarray(scenarios, dtype=object),
        feature_names,
    )


def generate_fma_audio_benchmark(
    *,
    audio_dir: str | Path,
    metadata_dir: str | Path,
    cache_dir: str | Path,
    clients: int = 8,
    max_tracks: int = 1200,
    max_train_pairs: int = 6000,
    max_test_pairs: int = 3000,
    negative_ratio: float = 1.0,
    sample_rate: int = 8000,
    max_seconds: float = 25.0,
    seed: int = 31,
) -> BenchmarkData:
    """Build an audio copy-detection benchmark from FMA-small MP3 files."""

    cache = _prepare_fma_audio_cache(
        audio_dir=audio_dir,
        metadata_dir=metadata_dir,
        cache_dir=cache_dir,
        max_tracks=max_tracks,
        sample_rate=sample_rate,
        max_seconds=max_seconds,
        seed=seed,
    )
    track_ids = cache["track_ids"]
    genres = cache["genres"].astype(str)
    client_raw = cache["clients"].astype(int) % max(clients, 1)
    descriptors = cache["descriptors"]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(track_ids))
    split = max(1, int(round(0.70 * len(order))))
    train_idx = order[:split]
    test_idx = order[split:]
    if len(test_idx) < 2:
        test_idx = train_idx[-max(2, len(train_idx) // 4) :]
        train_idx = train_idx[: -len(test_idx)]

    train_pairs = _sample_fma_pairs(
        train_idx,
        genres,
        rng,
        max_positive_pairs=max_train_pairs,
        negative_ratio=negative_ratio,
    )
    test_pairs = _sample_fma_pairs(
        test_idx,
        genres,
        rng,
        max_positive_pairs=max_test_pairs,
        negative_ratio=negative_ratio,
    )
    x_train, y_train, client_train, scenario_train, feature_names = _fma_pairs_to_features(
        train_pairs,
        descriptors,
        genres,
        client_raw,
        client_count=clients,
    )
    x_test, y_test, client_test, scenario_test, _ = _fma_pairs_to_features(
        test_pairs,
        descriptors,
        genres,
        client_raw,
        client_count=clients,
    )
    manifest = {
        "generator": "fma_small_audio_transform_features",
        "audio_dir": str(audio_dir),
        "metadata_dir": str(metadata_dir),
        "cache_path": cache["cache_path"],
        "clients": clients,
        "tracks_loaded": int(len(track_ids)),
        "max_tracks": max_tracks,
        "sample_rate": sample_rate,
        "max_seconds": max_seconds,
        "max_train_positive_pairs": max_train_pairs,
        "max_test_positive_pairs": max_test_pairs,
        "negative_ratio": negative_ratio,
        "seed": seed,
        "source": "FMA-small MP3 audio and FMA metadata",
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "transformations": FMA_AUDIO_TRANSFORMS,
        "split_policy": "track-identity-disjoint random split",
        "asset_overlap": int(len(set(train_idx) & set(test_idx))),
    }
    return BenchmarkData(
        x_train=x_train,
        y_train=y_train,
        client_train=client_train,
        scenario_train=scenario_train,
        x_test=x_test,
        y_test=y_test,
        client_test=client_test,
        scenario_test=scenario_test,
        feature_names=feature_names,
        manifest=manifest,
    )
