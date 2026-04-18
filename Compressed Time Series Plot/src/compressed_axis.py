from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TimeSegment:
    start: float
    end: float

    @property
    def width(self) -> float:
        return self.end - self.start


class TimeAxisCompressor:
    """Map multiple time segments into one compressed plotting axis."""

    def __init__(self, segments: Sequence[Tuple[float, float]], gap: float = 0.05) -> None:
        if not segments:
            raise ValueError("segments cannot be empty")
        parsed = [TimeSegment(float(a), float(b)) for a, b in segments]
        if any(seg.end <= seg.start for seg in parsed):
            raise ValueError("each segment must satisfy end > start")
        self._segments: List[TimeSegment] = parsed
        self._gap = float(gap)
        self._starts = self._build_compressed_starts()

    @property
    def segments(self) -> Sequence[TimeSegment]:
        return self._segments

    @property
    def gap(self) -> float:
        return self._gap

    @property
    def total_width(self) -> float:
        return self._starts[-1] + self._segments[-1].width

    @property
    def break_positions(self) -> List[float]:
        """Compressed x positions where axis breaks happen between segments."""
        positions: List[float] = []
        for i in range(len(self._segments) - 1):
            positions.append(self._starts[i] + self._segments[i].width + self._gap / 2)
        return positions

    def map_times(self, t: Iterable[float]) -> np.ndarray:
        """Return compressed-axis x values for times that lie inside configured segments."""
        arr = np.asarray(list(t), dtype=float)
        out = np.full_like(arr, np.nan, dtype=float)
        for idx, seg in enumerate(self._segments):
            mask = (arr >= seg.start) & (arr <= seg.end)
            out[mask] = self._starts[idx] + (arr[mask] - seg.start)
        return out

    def segment_compressed_bounds(self) -> List[Tuple[float, float]]:
        bounds: List[Tuple[float, float]] = []
        for i, seg in enumerate(self._segments):
            a = self._starts[i]
            b = a + seg.width
            bounds.append((a, b))
        return bounds

    def map_interval(self, start: float, end: float) -> Tuple[float, float]:
        """Map an interval fully contained in one segment into compressed coordinates."""
        for idx, seg in enumerate(self._segments):
            if seg.start <= start <= end <= seg.end:
                mapped_start = self._starts[idx] + (start - seg.start)
                mapped_end = self._starts[idx] + (end - seg.start)
                return mapped_start, mapped_end
        raise ValueError("interval is not fully contained in any configured segment")

    def _build_compressed_starts(self) -> List[float]:
        starts = [0.0]
        for i in range(1, len(self._segments)):
            prev = self._segments[i - 1]
            starts.append(starts[-1] + prev.width + self._gap)
        return starts
