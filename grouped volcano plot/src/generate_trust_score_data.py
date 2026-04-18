from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


GROUP_ORDER = ["Train Legit", "Test Legit", "Attacker 1", "Attacker 2"]
SAMPLE_START = 0.0
SAMPLE_END = 300.0
SAMPLE_COUNT = 300
REAL_GROUP_FILES = {
    "Train Legit": "theta_train_sampled.mat",
    "Test Legit": "theta_pool_legal.mat",
    "Attacker 1": "theta_pool_attack.mat",
}


def _clip(a: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(a, low, high)


def _sample_times(count: int) -> np.ndarray:
    return np.linspace(SAMPLE_START, SAMPLE_END, count)


def _build_group_frame(group: str, scores: np.ndarray) -> pd.DataFrame:
    sample_times = _sample_times(len(scores))
    return pd.DataFrame(
        {
            "group": group,
            "sample_id": [f"{group.replace(' ', '_').lower()}_{i+1}" for i in range(len(scores))],
            "sample_time": sample_times,
            "trust_score": scores,
        }
    )


def _normalize_vector(values: np.ndarray, *, source: Path) -> np.ndarray:
    array = np.asarray(values, dtype=float).squeeze()
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1:
        raise ValueError(f"Expected a vector in {source}, got shape {array.shape}")
    return array


def _normalize_to_count(values: np.ndarray, target_count: int) -> np.ndarray:
    if values.size >= target_count:
        return values[:target_count]
    pad_width = target_count - values.size
    return np.pad(values, (0, pad_width), mode="constant", constant_values=0.0)


def _extract_numeric_array(value: object) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for item in value.ravel():
                extracted = _extract_numeric_array(item)
                if extracted is not None:
                    return extracted
            return None
        if np.issubdtype(value.dtype, np.number):
            return np.asarray(value, dtype=float)
        return None

    if hasattr(value, "_fieldnames"):
        for field in value._fieldnames:
            extracted = _extract_numeric_array(getattr(value, field))
            if extracted is not None:
                return extracted
        return None

    if np.isscalar(value):
        return np.asarray([value], dtype=float)

    return None


def _load_mat_vector(mat_path: Path) -> np.ndarray:
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    payload = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    candidates: list[np.ndarray] = []
    for key in sorted(payload):
        if key.startswith("__"):
            continue
        extracted = _extract_numeric_array(payload[key])
        if extracted is not None:
            candidates.append(extracted)

    if not candidates:
        raise ValueError(f"No numeric vector found in {mat_path}")

    vector = _normalize_vector(candidates[0], source=mat_path)
    return _normalize_to_count(vector, SAMPLE_COUNT)


def _resolve_mat_path(mat_dir: Path, stem: str) -> Path:
    candidates = [mat_dir / stem, mat_dir / f"{stem}.mat"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {stem} or {stem}.mat under {mat_dir}")


def build_real_data(mat_dir: Path) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for group in ("Train Legit", "Test Legit", "Attacker 1"):
        mat_file = _resolve_mat_path(mat_dir, REAL_GROUP_FILES[group])
        scores = _load_mat_vector(mat_file)
        chunks.append(_build_group_frame(group, scores))

    att2 = _clip(np.random.default_rng(2026).normal(loc=-1.10, scale=0.70, size=110), -5.0, 0.8)
    att2 = np.concatenate([att2, np.array([-4.8, -4.5, -4.2])])
    chunks.append(
        pd.DataFrame(
            {
                "group": "Attacker 2",
                "sample_id": [f"attacker_2_{i+1}" for i in range(len(att2))],
                "sample_time": _sample_times(len(att2)),
                "trust_score": att2,
            }
        )
    )

    data = pd.concat(chunks, ignore_index=True)
    data["group"] = pd.Categorical(data["group"], categories=GROUP_ORDER, ordered=True)
    return data.sort_values("group", kind="stable").reset_index(drop=True)


def build_demo_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    train = _clip(rng.normal(loc=1.15, scale=0.45, size=SAMPLE_COUNT), -0.2, 3.2)
    test = _clip(rng.normal(loc=1.35, scale=0.50, size=SAMPLE_COUNT), -0.3, 3.1)
    att1 = _clip(rng.normal(loc=-1.75, scale=0.85, size=SAMPLE_COUNT), -5.0, 1.2)
    att2 = _clip(rng.normal(loc=-1.10, scale=0.70, size=110), -5.0, 0.8)

    # Preserve the 300-sample convention for the real source groups while keeping Attacker 2
    # unchanged because no dedicated input series is available yet.
    train[:3] = np.array([2.5, 2.7, 2.8])
    test[:3] = np.array([-0.2, 0.0, 0.2])
    att1[:3] = np.array([-4.6, -4.3, -4.0])
    att2 = np.concatenate([att2, np.array([-4.8, -4.5, -4.2])])

    chunks: list[pd.DataFrame] = [
        _build_group_frame("Train Legit", train),
        _build_group_frame("Test Legit", test),
        _build_group_frame("Attacker 1", att1),
        pd.DataFrame(
            {
                "group": "Attacker 2",
                "sample_id": [f"attacker_2_{i+1}" for i in range(len(att2))],
                "sample_time": _sample_times(len(att2)),
                "trust_score": att2,
            }
        ),
    ]

    data = pd.concat(chunks, ignore_index=True)
    data["group"] = pd.Categorical(data["group"], categories=GROUP_ORDER, ordered=True)
    data = data.sort_values("group", kind="stable").reset_index(drop=True)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo trust-score data (decoupled CSV)")
    parser.add_argument("--mode", choices=("demo", "real"), default="demo")
    parser.add_argument("--mat-dir", type=Path, default=Path("data"), help="Directory containing theta_*.mat files")
    parser.add_argument("--output", type=Path, default=Path("data/trust_score_data.csv"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "real":
        df = build_real_data(mat_dir=args.mat_dir)
    else:
        df = build_demo_data(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
