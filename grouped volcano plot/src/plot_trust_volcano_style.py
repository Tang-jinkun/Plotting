from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"


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
TIME_MIN = 0.0
TIME_MAX = 300.0
GROUP_GAP = 35.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw trust-score distribution as a volcano-style plot")
    parser.add_argument("--input", type=Path, default=Path("data/trust_score_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/trust_score_volcano_style.png"))
    parser.add_argument("--theta0", type=float, default=0.0, help="Lower trust threshold")
    parser.add_argument("--thetaT", type=float, default=1.0, help="Upper trust threshold")
    parser.add_argument("--point-size", type=float, default=60.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--bg-alpha", type=float, default=0.05)
    parser.add_argument("--title", type=str, default="Trust Score Over Time by Group")
    parser.add_argument("--show-thresholds", action="store_true", default=True)
    parser.add_argument("--hide-thresholds", action="store_true")
    parser.add_argument("--show-counts", action="store_true", default=True)
    parser.add_argument("--hide-counts", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"group", "trust_score", "sample_time"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _darken_color(color: str, factor: float = 0.65) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


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
                markeredgecolor=_darken_color(GROUP_COLORS[g]),
                markeredgewidth=0.9,
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
        prop={"family": "Times New Roman"},
    )


def draw_plot(
    df: pd.DataFrame,
    output_path: Path,
    theta0: float,
    thetaT: float,
    point_size: float,
    alpha: float,
    bg_alpha: float,
    title: str,
    show_thresholds: bool,
    show_counts: bool,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    time_span = TIME_MAX - TIME_MIN
    group_offsets = {g: i * (time_span + GROUP_GAP) for i, g in enumerate(GROUP_ORDER)}

    y_min_data = float(df["trust_score"].min())
    y_max_data = float(df["trust_score"].max())
    y_span = y_max_data - y_min_data
    y_pad = max(0.25 * y_span * 0.2, 0.45)
    y_min = y_min_data - y_pad
    y_max = y_max_data + y_pad
    ax.set_ylim(y_min, y_max)

    x_min = group_offsets[GROUP_ORDER[0]] + TIME_MIN
    x_max = group_offsets[GROUP_ORDER[-1]] + TIME_MAX
    x_pad = max(time_span * 0.03, 6.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    high_y = y_max - 0.07 * max(y_max - y_min, 1.0)
    low_y = y_min + 0.07 * max(y_max - y_min, 1.0)

    for g in GROUP_ORDER:
        dg = df[df["group"] == g].copy()
        if dg.empty:
            continue
        x0 = group_offsets[g] + TIME_MIN
        x1 = group_offsets[g] + TIME_MAX
        ax.axvspan(x0, x1, color="#8f8f8f", alpha=bg_alpha, zorder=0)

        dg = dg.sort_values("sample_time", kind="stable")
        x_values = group_offsets[g] + np.clip(dg["sample_time"].to_numpy(dtype=float), TIME_MIN, TIME_MAX)
        ax.scatter(
            x_values,
            dg["trust_score"],
            s=point_size,
            c=GROUP_COLORS[g],
            marker=GROUP_MARKERS[g],
            edgecolors=[_darken_color(GROUP_COLORS[g])],
            linewidths=0.9,
            alpha=alpha,
            zorder=3,
        )

        # Keep the data faithful while still showing trend continuity over time.
        ax.plot(
            x_values,
            dg["trust_score"],
            color=GROUP_COLORS[g],
            linewidth=0.7,
            linestyle=(0, (3, 3)),
            alpha=0.9,
            zorder=2,
        )

        if show_counts:
            high = int((dg["trust_score"] > thetaT).sum())
            low = int((dg["trust_score"] < theta0).sum())
            x_center = group_offsets[g] + 0.5 * (TIME_MIN + TIME_MAX)
            ax.text(
                x_center,
                high_y,
                f"High: {high}",
                ha="center",
                va="top",
                fontsize=11,
                color="#1f5f1f",
                fontweight="bold",
                zorder=5,
                fontfamily="Times New Roman",
            )
            ax.text(
                x_center,
                low_y,
                f"Low: {low}",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#5e1f3a",
                fontweight="bold",
                zorder=5,
                fontfamily="Times New Roman",
            )

        ax.text(
            group_offsets[g] + 0.5 * (TIME_MIN + TIME_MAX),
            1.01,
            g,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#2a2a2a",
            fontfamily="Times New Roman",
        )

    if show_thresholds:
        ax.axhline(theta0, color="#b51e3f", linestyle=(0, (4, 3)), linewidth=1.8, zorder=2)
        ax.axhline(thetaT, color="#57bf23", linestyle=(0, (4, 3)), linewidth=1.8, zorder=2)
        x_annot = x_max + 0.35 * x_pad
        ax.text(x_annot, theta0 + 0.03, r"$\theta_0$", color="#b51e3f", fontsize=12, va="bottom", ha="left", fontfamily="Times New Roman")
        ax.text(x_annot, thetaT + 0.03, r"$\theta_T$", color="#57bf23", fontsize=12, va="bottom", ha="left", fontfamily="Times New Roman")

    xticks: list[float] = []
    xlabels: list[str] = []
    for g in GROUP_ORDER:
        for t in (0, 100, 200, 300):
            xticks.append(group_offsets[g] + t)
            xlabels.append(str(t))

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    for i in range(len(GROUP_ORDER) - 1):
        boundary = group_offsets[GROUP_ORDER[i]] + TIME_MAX + 0.5 * GROUP_GAP
        ax.axvline(boundary, color="#666666", linestyle=(0, (3, 3)), linewidth=0.9, alpha=0.7, zorder=1)

    ax.set_xlabel("Sample Time", fontsize=13, fontfamily="Times New Roman")
    ax.set_ylabel("Trust Score", fontsize=14, fontfamily="Times New Roman")
    fig.suptitle(title, fontsize=16, y=0.975, fontfamily="Times New Roman")
    fig.text(
        0.5,
        0.94,
        r"Grouped view: each group uses its own 0-300 time range; High: score > $\theta_T$, Low: score < $\theta_0$",
        ha="center",
        va="center",
        fontsize=12,
        fontfamily="Times New Roman",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
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
        bg_alpha=args.bg_alpha,
        title=args.title,
        show_thresholds=show_thresholds,
        show_counts=show_counts,
        dpi=args.dpi,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
