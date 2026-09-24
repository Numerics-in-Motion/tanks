"""Recompute and compare this ideal-tank study with its frozen canonical data.

Usage: python reproduce.py --quick  (four primary period pairs)
       python reproduce.py          (all 64 registered cells)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "solver"))

from tank_slosh049_divider import LABELS, LENGTHS, PERIODS, T3, T6  # noqa: E402
from tank_slosh049_exact import conservative_basis, ode_control, peak, table  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check(ref: dict, quick: bool) -> None:
    for module, want in ref["source_sha256"].items():
        got = sha(HERE / "solver" / (module + ".py"))
        if got != want:
            raise AssertionError(f"source changed: {module}")
    tol = ref["tolerances"]
    assert abs(T3 - ref["T3_s"]) < tol["period_abs_s"]
    assert abs(T6 - ref["T6_s"]) < tol["period_abs_s"]
    error = ode_control(3.0, T3, 8.0, 0.0, n_max=31)
    assert error < 1e-7, f"independent ODE disagrees by {error} mm"
    if quick:
        computed = []
        for label, period in zip(LABELS, PERIODS):
            for length in LENGTHS:
                rise, when = peak(length, period, 8.0, 0.0)
                computed.append(dict(protocol="fixed8", zeta=0.0, label=label,
                                     period_s=period, duration_s=8.0,
                                     length_m=length, rise_mm=1000*rise,
                                     peak_at_s=when,
                                     **conservative_basis(length, period, 8.0, 0.0)))
    else:
        computed = table()
    wanted = ref["cells"]
    if quick:
        wanted = [r for r in wanted if r["protocol"] == "fixed8" and r["zeta"] == 0.0]
    assert len(computed) == len(wanted)
    largest_rise = largest_time = 0.0
    for got, want in zip(computed, wanted):
        for name in ("protocol", "zeta", "label", "length_m", "duration_s"):
            assert got[name] == want[name], (name, got[name], want[name])
        dr = abs(got["rise_mm"] - want["rise_mm"])
        dt = abs(got["peak_at_s"] - want["peak_at_s"])
        largest_rise = max(largest_rise, dr)
        largest_time = max(largest_time, dt)
        assert dr < tol["rise_abs_mm"] and dt < tol["peak_time_abs_s"]
        assert got["eta_over_h_upper_bound"] < 0.10
        assert got["slope_upper_bound"] < 0.10
        assert got["freeboard_lower_bound_m"] > 0.20
    def pair(protocol: str, zeta: float, label: str):
        rows = [r for r in computed if r["protocol"] == protocol and r["zeta"] == zeta
                and r["label"] == label]
        return {r["length_m"]: r["rise_mm"] for r in rows}
    for zeta in ((0.0,) if quick else (0.0, 0.005, 0.02, 0.05)):
        p3 = pair("fixed8", zeta, "T3")
        p6 = pair("fixed8", zeta, "T6")
        assert p3[3.0] > p3[6.0]
        assert p6[3.0] < p6[6.0]
    print(f"OK: {len(computed)} cells; max rise difference {largest_rise:.3g} mm; "
          f"max peak-time difference {largest_time:.3g} s; ODE control {error:.3g} mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    check(json.loads((HERE / "reference" / "reference_canonical.json").read_text(encoding="utf-8")),
          args.quick)
