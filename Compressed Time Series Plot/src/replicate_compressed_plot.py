from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from compressed_axis import TimeAxisCompressor


@dataclass(frozen=True)
class PlotConfig:
    segments: Sequence[Tuple[float, float]]
    attack_windows: Sequence[Tuple[float, float]]
    scenario_spans: Sequence[Tuple[float, float, str]]
    xticks_real: Sequence[float]
    gap: float = 0.045
    y_lim: Tuple[float, float] = (-35.0, 35.0)


def build_demo_series(t: np.ndarray, seed: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build synthetic top/bottom series with behavior close to the target figure."""
    rng = np.random.default_rng(seed)

    u_large = np.zeros_like(t)
    u_small = np.zeros_like(t)
    a_large = np.zeros_like(t)
    a_small = np.zeros_like(t)

    # Scenario 1: smooth low-frequency behavior with pulses near attack slots.
    s1 = (t >= 0.08) & (t <= 0.31)
    tau = t[s1]
    base1 = -5.0 - 8.0 * np.sin((tau - 0.08) * 32.0) - 10.0 * np.exp(-((tau - 0.24) ** 2) / 0.001)
    pulse1 = 12.0 * np.exp(-((tau - 0.095) ** 2) / 0.000005)
    pulse2 = 10.0 * np.exp(-((tau - 0.198) ** 2) / 0.000006)
    pulse3 = 11.0 * np.exp(-((tau - 0.299) ** 2) / 0.000006)
    u_large[s1] = base1 + pulse1 + pulse2 + pulse3
    u_small[s1] = base1 + 0.8 * (pulse1 + pulse2 + pulse3)

    # Scenario 2: noisy higher-amplitude region.
    s2 = (t >= 4.6) & (t <= 4.9)
    tau = t[s2]
    trend2 = 8.0 * np.sin((tau - 4.65) * 15.0) + 5.0
    u_large[s2] = trend2 + rng.normal(0.0, 8.0, size=tau.size)
    u_small[s2] = trend2 + rng.normal(0.0, 4.0, size=tau.size)

    # Scenario 3: noisy region around zero with smaller mean value.
    s3a = (t >= 6.9) & (t <= 7.0)
    tau = t[s3a]
    u_large[s3a] = rng.normal(2.0, 8.0, size=tau.size)
    u_small[s3a] = rng.normal(2.0, 4.0, size=tau.size)

    s3b = (t >= 7.75) & (t <= 7.98)
    tau = t[s3b]
    u_large[s3b] = rng.normal(4.0, 9.0, size=tau.size)
    u_small[s3b] = rng.normal(4.0, 4.5, size=tau.size)

    # Bottom panel: smaller-magnitude responses.
    a_large = 0.45 * u_large + 0.4 * np.sin(3.0 * t)
    a_small = 0.45 * u_small

    # Keep long unobserved regions flat so only configured segments are drawn after mapping.
    return u_large, u_small, a_large, a_small


def attack_profile(t: np.ndarray, windows: Sequence[Tuple[float, float]], low: float = -30.0, high: float = 30.0) -> np.ndarray:
    out = np.full_like(t, low, dtype=float)
    for start, end in windows:
        out[(t >= start) & (t <= end)] = high
    return out


def plot_segments(
    ax: plt.Axes,
    compressor: TimeAxisCompressor,
    t: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.4,
    label: str | None = None,
) -> None:
    first = True
    for seg in compressor.segments:
        mask = (t >= seg.start) & (t <= seg.end)
        if not np.any(mask):
            continue
        x_seg = compressor.map_times(t[mask])
        ax.plot(
            x_seg,
            y[mask],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label if first else None,
        )
        first = False


def add_break_marks(ax: plt.Axes, x_breaks: Sequence[float], size: float = 0.012) -> None:
    for xb in x_breaks:
        kwargs = dict(transform=ax.get_xaxis_transform(), color="black", clip_on=False, linewidth=1.0)
        ax.plot([xb - size, xb + size], [-0.02, 0.04], **kwargs)
        ax.plot([xb - size + 0.012, xb + size + 0.012], [-0.02, 0.04], **kwargs)


def add_attack_window_boxes(
    ax: plt.Axes,
    compressor: TimeAxisCompressor,
    windows: Sequence[Tuple[float, float]],
    y_min: float,
    y_max: float,
) -> None:
    for start, end in windows:
        x0, x1 = compressor.map_interval(start, end)
        rect = Rectangle(
            (x0, y_min),
            x1 - x0,
            y_max - y_min,
            fill=False,
            edgecolor="black",
            linestyle=":",
            linewidth=0.9,
            zorder=0,
        )
        ax.add_patch(rect)


def plot_series_with_compressed_axis(config: PlotConfig, output_path: Path, dpi: int = 300) -> None:
    compressor = TimeAxisCompressor(config.segments, gap=config.gap)

    t = np.linspace(min(s[0] for s in config.segments), max(s[1] for s in config.segments), 2600)
    u_large, u_small, a_large, a_small = build_demo_series(t)
    attack = attack_profile(t, config.attack_windows)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 6.8), sharex=True, gridspec_kw={"hspace": 0.08})

    # Top panel
    plot_segments(ax1, compressor, t, attack, color="black", linestyle=(0, (1, 1)), linewidth=1.0, label="Attack slot")
    plot_segments(ax1, compressor, t, u_large, color="#6b8e23", linestyle="-", linewidth=1.4, label="FDI: Large amplitude")
    plot_segments(ax1, compressor, t, u_small, color="#b05a87", linestyle=(0, (4, 4)), linewidth=1.2, label="FDI: Small amplitude")

    # Bottom panel
    plot_segments(ax2, compressor, t, attack, color="black", linestyle=(0, (1, 1)), linewidth=1.0)
    plot_segments(ax2, compressor, t, a_large, color="#6b8e23", linestyle="-", linewidth=1.4)
    plot_segments(ax2, compressor, t, a_small, color="#b05a87", linestyle=(0, (4, 4)), linewidth=1.2)

    for ax in (ax1, ax2):
        add_attack_window_boxes(ax, compressor, config.attack_windows, config.y_lim[0] + 5.0, config.y_lim[1] - 5.0)
        add_break_marks(ax, compressor.break_positions)
        ax.set_ylim(*config.y_lim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Scenario labels and separators
    for _, right, _ in config.scenario_spans[:-1]:
        x_sep = compressor.map_times([right])[0] + compressor.gap / 2
        ax1.axvline(x=x_sep, color="black", linestyle=(0, (4, 4)), linewidth=1.0)
        ax2.axvline(x=x_sep, color="black", linestyle=(0, (4, 4)), linewidth=1.0)

    for left, right, label in config.scenario_spans:
        x_left = compressor.map_times([left])[0]
        x_right = compressor.map_times([right])[0]
        ax1.text((x_left + x_right) / 2, config.y_lim[1] - 1.5, label, ha="center", va="top", fontsize=12)

    mapped_ticks = compressor.map_times(config.xticks_real)
    valid = ~np.isnan(mapped_ticks)
    ax2.set_xticks(mapped_ticks[valid])
    ax2.set_xticklabels([f"{x:g}" for x, ok in zip(config.xticks_real, valid) if ok])

    ax1.set_ylabel(r"MSM-SSS: $u(t)$", fontsize=14)
    ax2.set_ylabel(r"TARSC: $a(t)$", fontsize=14)
    ax2.set_xlabel("Time (s)", fontsize=14)

    ax1.legend(loc="upper center", ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.18), handlelength=3.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate a compressed time-series figure with broken x-axis.")
    parser.add_argument("--output", type=Path, default=Path("output/compressed_time_series_demo.png"), help="Output image path")
    parser.add_argument("--dpi", type=int, default=300, help="Image DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PlotConfig(
        segments=[(0.08, 0.31), (4.60, 4.90), (6.90, 7.00), (7.75, 7.98)],
        attack_windows=[(0.09, 0.102), (0.19, 0.202), (0.299, 0.31), (4.60, 4.90), (6.90, 7.00), (7.75, 7.98)],
        scenario_spans=[(0.08, 0.31, "Scenario 1"), (4.60, 4.90, "Scenario 2"), (6.90, 7.98, "Scenario 3")],
        xticks_real=[0.1, 0.2, 0.3, 4.6, 4.7, 4.8, 4.9, 7.0, 7.8, 7.9],
        gap=0.05,
        y_lim=(-35.0, 35.0),
    )

    plot_series_with_compressed_axis(config, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
