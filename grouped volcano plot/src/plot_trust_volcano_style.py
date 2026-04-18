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
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--jitter", type=float, default=0.33)
    parser.add_argument("--bg-alpha", type=float, default=0.05)
    parser.add_argument("--title", type=str, default="Trust Score Distribution by Group")
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


def _make_legend(fig: plt.Figure, show_thresholds: bool) -> None:
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

    fig.legend(
        handles=handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        frameon=True,
        fancybox=False,
        edgecolor="#555555",
        fontsize=10,
    )


def draw_plot(
    df: pd.DataFrame,
    output_path: Path,
    theta0: float,
    thetaT: float,
    point_size: float,
    alpha: float,
    jitter: float,
    bg_alpha: float,
    title: str,
    show_thresholds: bool,
    show_counts: bool,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.0))

    x_map = {g: i + 1 for i, g in enumerate(GROUP_ORDER)}
    rng = np.random.default_rng(2026)

    y_min_data = float(df["trust_score"].min())
    y_max_data = float(df["trust_score"].max())
    y_span = y_max_data - y_min_data
    y_pad = max(0.25 * y_span * 0.2, 0.45)
    y_min = y_min_data - y_pad
    y_max = y_max_data + y_pad
    ax.set_ylim(y_min, y_max)

    high_y = y_max - 0.45
    low_y = y_min + 0.45

    for g in GROUP_ORDER:
        x = x_map[g]
        ax.axvspan(x - 0.45, x + 0.45, color="#8f8f8f", alpha=bg_alpha, zorder=0)

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
            high = int((dg["trust_score"] > thetaT).sum())
            low = int((dg["trust_score"] < theta0).sum())
            ax.text(
                x_map[g],
                high_y,
                f"High: {high}",
                ha="center",
                va="top",
                fontsize=11,
                color="#1f5f1f",
                fontweight="bold",
                zorder=5,
            )
            ax.text(
                x_map[g],
                low_y,
                f"Low: {low}",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#5e1f3a",
                fontweight="bold",
                zorder=5,
            )

    if show_thresholds:
        ax.axhline(theta0, color="#b51e3f", linestyle=(0, (4, 3)), linewidth=1.8, zorder=2)
        ax.axhline(thetaT, color="#57bf23", linestyle=(0, (4, 3)), linewidth=1.8, zorder=2)
        ax.text(4.49, theta0 + 0.03, r"$\theta_0$", color="#b51e3f", fontsize=12, va="bottom", ha="left")
        ax.text(4.49, thetaT + 0.03, r"$\theta_T$", color="#57bf23", fontsize=12, va="bottom", ha="left")

    ax.set_xlim(0.5, len(GROUP_ORDER) + 0.62)
    ax.set_xticks([x_map[g] for g in GROUP_ORDER])
    ax.set_xticklabels(GROUP_ORDER, fontsize=12)
    ax.set_xlabel("Group", fontsize=13)
    ax.set_ylabel("Trust Score", fontsize=14)
    fig.suptitle(title, fontsize=16, y=0.975)
    fig.text(
        0.5,
        0.94,
        r"High: score > $\theta_T$, Low: score < $\theta_0$",
        ha="center",
        va="center",
        fontsize=13,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=5)
    ax.grid(axis="y", alpha=0.15)

    _make_legend(fig, show_thresholds=show_thresholds)
    plt.tight_layout(rect=[0, 0, 1, 0.78])
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
        bg_alpha=args.bg_alpha,
        title=args.title,
        show_thresholds=show_thresholds,
        show_counts=show_counts,
        dpi=args.dpi,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
