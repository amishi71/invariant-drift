"""Per-event throughput/latency and memory footprint for each detector.

Exists because the abstract makes a specific, previously-unmeasured claim:
"triggers process millions of events per second." A detector that's
statistically excellent but takes 50 microseconds/event is not deployable
in that regime -- this module puts an actual number against the claim
instead of leaving it asserted.

Same number matters to two different embedded-constraint audiences at
once, which is worth being explicit about rather than framing this as
HEP-only: an FPGA-based L1 trigger and a radiation-hardened, low-power
spacecraft flight computer are both hard-constrained on cycles/event and
memory footprint, for different but structurally similar reasons (L1:
fixed-latency budget measured in tens of nanoseconds to low microseconds;
flight computer: strict power/thermal envelope with no serviceable
hardware). Reporting both wall-clock throughput and peak memory per
detector instance lets a reader in either context read off what they
actually need without a second benchmark.

This is deliberately NOT a claim about actual FPGA feasibility -- these
are Python/NumPy timings on a general-purpose CPU, not synthesized HDL
cycle counts. It answers "which of these detectors is cheap enough to be
worth prototyping for a hard-real-time target," not "this runs on an
FPGA." Say so explicitly wherever these numbers are reported.
"""

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class ThroughputResult:
    detector_name: str
    n_events: int
    total_time_sec: float
    mean_us_per_event: float
    p99_us_per_event: float
    peak_memory_kb: float


def benchmark_detector(
    detector_name: str,
    detector_factory: Callable[[], object],
    residuals: np.ndarray,
    n_warmup: int = 100,
) -> ThroughputResult:
    """Times `update()` + `evaluate_drift()` per event for a fresh detector
    instance streamed over `residuals`. `n_warmup` events are run and
    discarded before timing starts (JIT/allocator warm-up; river's
    detectors in particular allocate internal state lazily on first use).
    """
    detector = detector_factory()
    residuals = np.asarray(residuals, dtype=np.float64)

    for r in residuals[:n_warmup]:
        detector.update(float(r))
        if detector.is_ready():
            detector.evaluate_drift()

    remaining = residuals[n_warmup:]
    per_event_sec = np.empty(len(remaining), dtype=np.float64)

    tracemalloc.start()
    t_start = time.perf_counter()
    for idx, r in enumerate(remaining):
        t0 = time.perf_counter()
        detector.update(float(r))
        if detector.is_ready():
            detector.evaluate_drift()
        per_event_sec[idx] = time.perf_counter() - t0
    total_time = time.perf_counter() - t_start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    per_event_us = per_event_sec * 1e6
    return ThroughputResult(
        detector_name=detector_name,
        n_events=len(remaining),
        total_time_sec=total_time,
        mean_us_per_event=float(np.mean(per_event_us)),
        p99_us_per_event=float(np.percentile(per_event_us, 99)),
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
        f"{'detector':<14} {'mean us/event':>14} {'p99 us/event':>14} {'peak KB':>10}",
        "-" * 56,
    ]
    for r in results:
        lines.append(
            f"{r.detector_name:<14} {r.mean_us_per_event:>14.2f} "
            f"{r.p99_us_per_event:>14.2f} {r.peak_memory_kb:>10.1f}"
        )
    return "\n".join(lines)
