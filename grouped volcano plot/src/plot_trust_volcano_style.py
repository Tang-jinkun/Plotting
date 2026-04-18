from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


GROUP_ORDER = ["Train Legit", "Test Legit", "Attacker 1", "Attacker 2"]
GROUP_COLORS = {
    "Train Legit": "#00e51a",
    "Test Legit": "#9ad27c",
    "Attacker 1": "#e58d61",
    "Attacker 2": "#d98ae0",
}
GROUP_MARKERS = {
    "Train Legit": "o",
    "Test Legit": "D",
    "Attacker 1": "^",
    "Attacker 2": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw trust-score distribution as a volcano-style plot")
    parser.add_argument("--input", type=Path, default=Path("data/trust_score_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/trust_score_volcano_style.png"))
    parser.add_argument("--theta0", type=float, default=0.0, help="Lower trust threshold")
    parser.add_argument("--thetaT", type=float, default=1.0, help="Upper trust threshold")
    parser.add_argument("--point-size", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--jitter", type=float, default=0.33)
    parser.add_argument("--show-thresholds", action="store_true", default=True)
    parser.add_argument("--hide-thresholds", action="store_true")
    parser.add_argument("--show-counts", action="store_true", default=True)
    parser.add_argument("--hide-counts", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"group", "trust_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _make_legend(ax: plt.Axes, show_thresholds: bool) -> None:
    handles: list[Line2D] = []
    for g in GROUP_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=GROUP_MARKERS[g],
                color="none",
                markerfacecolor=GROUP_COLORS[g],
                markeredgecolor="#1f1f1f",
                markeredgewidth=0.8,
                linestyle="None",
                markersize=8,
                label=g,
            )
        )

    if show_thresholds:
        handles.extend(
            [
                Line2D([0], [0], color="#b51e3f", lw=2.2, linestyle=(0, (4, 3)), label=r"Threshold: $\theta_0$"),
                Line2D([0], [0], color="#57bf23", lw=2.2, linestyle=(0, (4, 3)), label=r"Threshold: $\theta_T$"),
            ]
        )

    ax.legend(
        handles=handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=True,
        fancybox=False,
        edgecolor="#555555",
        fontsize=11,
    )


def draw_plot(
    df: pd.DataFrame,
    output_path: Path,
    theta0: float,
    thetaT: float,
    point_size: float,
    alpha: float,
    jitter: float,
    show_thresholds: bool,
    show_counts: bool,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.0))

    x_map = {g: i + 1 for i, g in enumerate(GROUP_ORDER)}
    rng = np.random.default_rng(2026)

    y_min = float(df["trust_score"].min())
    y_max = float(df["trust_score"].max())
    pad = max((y_max - y_min) * 0.12, 0.5)
    ax.set_ylim(y_min - pad, y_max + pad)

    for g in GROUP_ORDER:
        x = x_map[g]
        ax.axvspan(x - 0.45, x + 0.45, color="#efefef", alpha=0.7, zorder=0)

    for g in GROUP_ORDER:
        dg = df[df["group"] == g].copy()
        if dg.empty:
            continue
        xs = x_map[g] + rng.uniform(-jitter, jitter, len(dg))
        ax.scatter(
            xs,
            dg["trust_score"],
            s=point_size,
            c=GROUP_COLORS[g],
            marker=GROUP_MARKERS[g],
            edgecolors="#242424",
            linewidths=0.6,
            alpha=alpha,
            zorder=3,
        )

        if show_counts:
            high = int((dg["trust_score"] >= thetaT).sum())
            low = int((dg["trust_score"] <= theta0).sum())
            ax.text(
                x_map[g],
                dg["trust_score"].max() + pad * 0.20,
                f"High: {high}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#1f5f1f",
                fontweight="bold",
                zorder=5,
            )
            ax.text(
                x_map[g],
                dg["trust_score"].min() - pad * 0.20,
                f"Low: {low}",
                ha="center",
                va="top",
                fontsize=10,
                color="#5e1f3a",
                fontweight="bold",
                zorder=5,
            )

    if show_thresholds:
        ax.axhline(theta0, color="#b51e3f", linestyle=(0, (4, 3)), linewidth=2.2, zorder=2)
        ax.axhline(thetaT, color="#57bf23", linestyle=(0, (4, 3)), linewidth=2.2, zorder=2)
        ax.text(4.46, theta0 + 0.02, r"$\theta_0$", color="#b51e3f", fontsize=12, va="bottom", ha="left")
        ax.text(4.46, thetaT + 0.02, r"$\theta_T$", color="#57bf23", fontsize=12, va="bottom", ha="left")

    ax.set_xlim(0.5, len(GROUP_ORDER) + 0.5)
    ax.set_xticks([x_map[g] for g in GROUP_ORDER])
    ax.set_xticklabels(GROUP_ORDER, fontsize=12)
    ax.set_xlabel("Group", fontsize=13)
    ax.set_ylabel("Trust Score", fontsize=14)
    ax.set_title("Trust Score Distribution as Volcano-style Plot", fontsize=15, pad=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=5)

    _make_legend(ax, show_thresholds=show_thresholds)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    show_thresholds = args.show_thresholds and not args.hide_thresholds
    show_counts = args.show_counts and not args.hide_counts

    df = pd.read_csv(args.input)
    _validate_columns(df)

    # Keep plotting order stable when user-provided categories follow expected names.
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    df = df.sort_values("group", kind="stable").reset_index(drop=True)

    draw_plot(
        df=df,
        output_path=args.output,
        theta0=args.theta0,
        thetaT=args.thetaT,
        point_size=args.point_size,
        alpha=args.alpha,
        jitter=args.jitter,
        show_thresholds=show_thresholds,
        show_counts=show_counts,
        dpi=args.dpi,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
