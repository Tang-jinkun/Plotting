from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = ["group1", "group2", "group3", "group4", "group5", "group6"]

SPECIAL_GENES = {
    "group1": ["Peg10", "Ldhc", "Nppa"],
    "group2": ["Slc14a2", "Igkv13-85", "Mael", "Mical2"],
    "group3": ["Lypd8", "Gsta13", "Cimip3", "Iqcf2", "Gata1"],
    "group4": ["Srl", "H2-Q8", "Rnf128", "ATP6", "Tnp1", "COX2"],
    "group5": ["Nnt", "Gal3st3", "Adcy4"],
    "group6": ["Crisp1", "Cfd", "Spink8", "Krt20", "Zg16", "Igha"],
}


def _make_group_data(group: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    # Group-specific center shifts mimic the different vertical distributions in the target figure.
    group_shift = {
        "group1": -0.6,
        "group2": -1.1,
        "group3": -0.3,
        "group4": 0.1,
        "group5": 0.0,
        "group6": 0.4,
    }[group]

    log2_fc = rng.normal(loc=group_shift, scale=1.45, size=n)
    log2_fc = np.clip(log2_fc, -7.7, 7.7)

    neg_log10_p = rng.gamma(shape=1.5, scale=1.0, size=n)
    neg_log10_p = np.clip(neg_log10_p, 0.02, 8.0)
    p_adj = 10 ** (-neg_log10_p)

    genes = [f"{group}_gene_{i + 1}" for i in range(n)]
    return pd.DataFrame(
        {
            "group": group,
            "gene": genes,
            "log2_fc": log2_fc,
            "p_adj": p_adj,
        }
    )


def _inject_special_genes(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    for group, names in SPECIAL_GENES.items():
        idx = out.index[out["group"] == group]
        replace_idx = rng.choice(idx, size=len(names), replace=False)
        for i, row_idx in enumerate(replace_idx):
            name = names[i]
            out.loc[row_idx, "gene"] = name
            if i < len(names) // 2:
                out.loc[row_idx, "log2_fc"] = rng.uniform(2.3, 7.4)
                out.loc[row_idx, "p_adj"] = 10 ** (-rng.uniform(2.0, 4.0))
            else:
                out.loc[row_idx, "log2_fc"] = -rng.uniform(2.3, 7.4)
                out.loc[row_idx, "p_adj"] = 10 ** (-rng.uniform(2.0, 4.0))
    return out


def build_demo_data(points_per_group: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_parts = [_make_group_data(g, points_per_group, rng) for g in GROUPS]
    data = pd.concat(all_parts, ignore_index=True)
    data = _inject_special_genes(data, rng)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate example data for grouped volcano plot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/volcano_data.csv"),
        help="Output CSV file path",
    )
    parser.add_argument("--points-per-group", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_demo_data(points_per_group=args.points_per_group, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Saved {len(data)} rows to {args.output}")


if __name__ == "__main__":
    main()
