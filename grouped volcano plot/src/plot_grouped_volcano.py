from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Times New Roman"


GROUP_ORDER = ["group1", "group2", "group3", "group4", "group5", "group6"]
GROUP_COLORS = {
    "group1": "#9ecae1",
    "group2": "#b2df8a",
    "group3": "#f4a3a8",
    "group4": "#f4c77b",
    "group5": "#c9bfdc",
    "group6": "#f28e8a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw grouped volcano plot from a CSV file")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/volcano_data.csv"),
        help="Input CSV with columns: group,gene,log2_fc,p_adj",
    )
    parser.add_argument("--output", type=Path, default=Path("output/grouped_volcano_plot.png"))
    parser.add_argument("--fc-threshold", type=float, default=1.0)
    parser.add_argument("--padj-threshold", type=float, default=0.05)
    parser.add_argument("--labels-per-side", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"group", "gene", "log2_fc", "p_adj"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _scale_sizes(neg_log10_p: pd.Series) -> pd.Series:
    return 18 + 28 * neg_log10_p


def _pick_labels(
    d: pd.DataFrame,
    fc_threshold: float,
    padj_threshold: float,
    labels_per_side: int,
) -> pd.DataFrame:
    sig = d[(d["p_adj"] < padj_threshold) & (d["log2_fc"].abs() >= fc_threshold)].copy()
    up = sig[sig["log2_fc"] > 0].nlargest(labels_per_side, "log2_fc")
    down = sig[sig["log2_fc"] < 0].nsmallest(labels_per_side, "log2_fc")
    return pd.concat([up, down], ignore_index=True)


def draw_plot(
    df: pd.DataFrame,
    output_path: Path,
    fc_threshold: float,
    padj_threshold: float,
    labels_per_side: int,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.5))

    ymin, ymax = -8.0, 8.0
    ax.set_ylim(ymin, ymax)

    band_bottom, band_height = -0.85, 1.7

    for i, group in enumerate(GROUP_ORDER, start=1):
        x_center = float(i)

        ax.add_patch(
            Rectangle(
                (x_center - 0.46, ymin + 0.3),
                0.92,
                ymax - ymin - 1.2,
                facecolor="#ececec",
                edgecolor="none",
                alpha=0.7,
                zorder=0,
            )
        )

        ax.add_patch(
            Rectangle(
                (x_center - 0.5, band_bottom),
                1.0,
                band_height,
                facecolor=GROUP_COLORS[group],
                edgecolor="none",
                alpha=1.0,
                zorder=4,
            )
        )
        ax.text(
            x_center,
            -0.05,
            group,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#1b2a34" if group != "group4" else "#3143aa",
            zorder=5,
            fontfamily="Times New Roman",
        )

    for i, group in enumerate(GROUP_ORDER, start=1):
        dg = df[df["group"] == group].copy()
        if dg.empty:
            continue

        jitter = np.random.default_rng(2026 + i).uniform(-0.38, 0.38, size=len(dg))
        dg["x"] = i + jitter
        dg["neg_log10_p"] = -np.log10(np.clip(dg["p_adj"], 1e-300, 1.0))
        dg["size"] = _scale_sizes(dg["neg_log10_p"])

        ax.scatter(
            dg["x"],
            dg["log2_fc"],
            s=dg["size"],
            c=GROUP_COLORS[group],
            edgecolors="#444444",
            linewidths=0.8,
            alpha=0.55,
            zorder=3,
        )

        up_count = int(((dg["log2_fc"] >= fc_threshold) & (dg["p_adj"] < padj_threshold)).sum())
        down_count = int(((dg["log2_fc"] <= -fc_threshold) & (dg["p_adj"] < padj_threshold)).sum())

        ax.text(
            i,
            dg["log2_fc"].max() + 0.45,
            f"Up: {up_count}",
            color="#9e2f2f",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
            fontfamily="Times New Roman",
        )
        ax.text(
            i,
            dg["log2_fc"].min() - 0.55,
            f"Down: {down_count}",
            color="#2e356d",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
            fontfamily="Times New Roman",
        )

        labels = _pick_labels(dg, fc_threshold, padj_threshold, labels_per_side)
        for _, row in labels.iterrows():
            ax.text(
                row["x"] + 0.06,
                row["log2_fc"] + (0.22 if row["log2_fc"] >= 0 else -0.22),
                str(row["gene"]),
                fontsize=10,
                fontstyle="italic",
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "#f3f3f3",
                    "edgecolor": "#d0d0d0",
                    "linewidth": 0.8,
                    "alpha": 0.95,
                },
                zorder=7,
                fontfamily="Times New Roman",
            )

    legend_values = [0, 2, 4]
    legend_handles = [
        plt.scatter([], [], s=_scale_sizes(pd.Series([v])).iloc[0], color="#555555", alpha=0.9)
        for v in legend_values
    ]
    ax.legend(
        legend_handles,
        [str(v) for v in legend_values],
        title=r"-$\log_{10}(P\mathrm{-adjusted})$",
        scatterpoints=1,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=11,
        title_fontsize=14,
        prop={"family": "Times New Roman"},
    )

    ax.set_xlim(0.4, len(GROUP_ORDER) + 0.6)
    ax.set_xticks([])
    ax.set_yticks(np.arange(-8, 9, 4))
    ax.set_ylabel(r"$\log_2(\mathrm{fold\ change})$", fontsize=16, fontfamily="Times New Roman")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.tick_params(axis="y", labelsize=12, width=1.2, length=6)
    for label in ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    _validate_columns(df)
    draw_plot(
        df=df,
        output_path=args.output,
        fc_threshold=args.fc_threshold,
        padj_threshold=args.padj_threshold,
        labels_per_side=args.labels_per_side,
        dpi=args.dpi,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
