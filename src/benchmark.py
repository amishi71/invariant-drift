"""Per-event throughput/latency and memory footprint for each detector.

Exists because the abstract makes a specific, previously-unmeasured claim:
"triggers process millions of events per second." Reading that as a
detector budget, a million events per second means <= ~1 microsecond per
event. This module puts an actual number against that bar instead of
leaving it asserted.

WHAT THESE NUMBERS ARE (and are not). They are wall-clock timings of
`update()` + `evaluate_drift()` in Python/NumPy on a general-purpose CPU.
They include interpreter and library overhead (for example river's
pure-Python internals, or NumPy call overhead), so they say little about
the *algorithmic* cost of a detector and nothing about FPGA, ASIC or
radiation-hardened flight-computer feasibility. The two embedded audiences
(an FPGA-based L1 trigger with a fixed latency budget; a spacecraft flight
computer with a strict power envelope) care about cycles/event and memory
footprint of a hardware implementation, which would be established by
synthesis or a C/embedded port, not by this benchmark. The defensible
cross-detector comparison is asymptotic cost (O(1) per event for CUSUM and
Page-Hinkley; growth with retained run-length hypotheses for BOCPD; a
two-sample KS test per event for KSWIN), with these timings as an
order-of-magnitude illustration. Say so wherever the numbers are reported.

MEASUREMENT PROTOCOL. Three separate passes over fresh detector instances
so that no measurement contaminates another:
  1. throughput: one timer around the whole loop, no per-event timers and
     no tracemalloc (tracemalloc slows allocation-heavy code several-fold);
  2. tail latency: per-event timers (p99). Each sample includes ~0.1-0.3 us
     of timer overhead, so p99 here is an upper-ish bound, not a pure cost;
  3. memory: tracemalloc, started after warm-up, peak traced allocation
     during the streamed loop (transient allocation, not total footprint).
"""

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ThroughputResult:
    detector_name: str
    n_events: int
    total_time_sec: float
    mean_us_per_event: float
    p99_us_per_event: float
    peak_memory_kb: float


def _step(detector, r: float) -> None:
    detector.update(r)
    if detector.is_ready():
        detector.evaluate_drift()


def _fresh_primed(detector_factory, warm) -> object:
    """Fresh detector, warmed up on `warm` (allocator/lazy-state warm-up;
    river's detectors in particular allocate internal state on first use)."""
    detector = detector_factory()
    for r in warm:
        _step(detector, float(r))
    return detector


def benchmark_detector(
    detector_name: str,
    detector_factory: Callable[[], object],
    residuals: np.ndarray,
    n_warmup: int = 100,
) -> ThroughputResult:
    residuals = np.asarray(residuals, dtype=np.float64)
    warm, remaining = residuals[:n_warmup], residuals[n_warmup:]
    n = len(remaining)
    stream = [float(r) for r in remaining]  # pre-convert: time the detector, not np->float

    # Pass 1: throughput (single timer, no tracemalloc, no per-event timers).
    detector = _fresh_primed(detector_factory, warm)
    t_start = time.perf_counter()
    for r in stream:
        _step(detector, r)
    total_time = time.perf_counter() - t_start

    # Pass 2: tail latency (per-event timers).
    detector = _fresh_primed(detector_factory, warm)
    per_event_sec = np.empty(n, dtype=np.float64)
    for idx, r in enumerate(stream):
        t0 = time.perf_counter()
        _step(detector, r)
        per_event_sec[idx] = time.perf_counter() - t0

    # Pass 3: memory (tracemalloc started after warm-up).
    detector = _fresh_primed(detector_factory, warm)
    tracemalloc.start()
    for r in stream:
        _step(detector, r)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return ThroughputResult(
        detector_name=detector_name,
        n_events=n,
        total_time_sec=total_time,
        mean_us_per_event=total_time / n * 1e6,
        p99_us_per_event=float(np.percentile(per_event_sec * 1e6, 99)),
        peak_memory_kb=peak / 1024.0,
    )


def benchmark_all(
    detector_specs: dict, reference_data: np.ndarray, n_events: int = 5000, seed: int = 0,
) -> "list[ThroughputResult]":
    """Runs benchmark_detector for every (name -> factory) pair in
    `detector_specs` (the same DETECTOR_SPECS dict main.py already
    defines) over a shared synthetic residual stream, so the numbers are
    directly comparable across detectors.
    """
    rng = np.random.RandomState(seed)
    stream_residuals = rng.normal(0, 1, n_events)  # pure-noise stream: measures overhead, not detection behavior
    results = []
    for name, factory in detector_specs.items():
        results.append(benchmark_detector(
            name, lambda f=factory: f(reference_data), stream_residuals,
        ))
    return results


def format_results(results: "list[ThroughputResult]") -> str:
    lines = [
        f"{'detector':<14} {'mean us/event':>14} {'k events/s':>12} "
        f"{'p99 us/event':>14} {'peak KB':>10}",
        "-" * 70,
    ]
    for r in results:
        k_eps = 1e3 / r.mean_us_per_event if r.mean_us_per_event > 0 else float("inf")
        lines.append(
            f"{r.detector_name:<14} {r.mean_us_per_event:>14.2f} {k_eps:>12.1f} "
            f"{r.p99_us_per_event:>14.2f} {r.peak_memory_kb:>10.1f}"
        )
    lines.append("")
    lines.append("1M events/s needs <= ~1 us/event. Python timings: not a hardware-feasibility claim.")
    return "\n".join(lines)
