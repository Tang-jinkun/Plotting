from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_ORDER = ["Train Legit", "Test Legit", "Attacker 1", "Attacker 2"]


def _clip(a: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(a, low, high)


def build_demo_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    train = _clip(rng.normal(loc=1.15, scale=0.45, size=180), -0.2, 3.2)
    test = _clip(rng.normal(loc=1.35, scale=0.50, size=220), -0.3, 3.1)
    att1 = _clip(rng.normal(loc=-1.75, scale=0.85, size=120), -5.0, 1.2)
    att2 = _clip(rng.normal(loc=-1.10, scale=0.70, size=110), -5.0, 0.8)

    # Add sparse outliers so the demo visually resembles the reference's long tails.
    train = np.concatenate([train, np.array([2.5, 2.7, 2.8])])
    test = np.concatenate([test, np.array([-0.2, 0.0, 0.2])])
    att1 = np.concatenate([att1, np.array([-4.6, -4.3, -4.0])])
    att2 = np.concatenate([att2, np.array([-4.8, -4.5, -4.2])])

    chunks: list[pd.DataFrame] = []
    for group, scores in [
        ("Train Legit", train),
        ("Test Legit", test),
        ("Attacker 1", att1),
        ("Attacker 2", att2),
    ]:
        chunks.append(
            pd.DataFrame(
                {
                    "group": group,
                    "sample_id": [f"{group.replace(' ', '_').lower()}_{i+1}" for i in range(len(scores))],
                    "trust_score": scores,
                }
            )
        )

    data = pd.concat(chunks, ignore_index=True)
    data["group"] = pd.Categorical(data["group"], categories=GROUP_ORDER, ordered=True)
    data = data.sort_values("group", kind="stable").reset_index(drop=True)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo trust-score data (decoupled CSV)")
    parser.add_argument("--output", type=Path, default=Path("data/trust_score_data.csv"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_demo_data(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
