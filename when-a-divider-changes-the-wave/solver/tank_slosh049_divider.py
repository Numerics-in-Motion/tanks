"""Registered Spinoff 049 center-divider screening and reproducible results."""
from __future__ import annotations

import argparse
import json

import numpy as np

from tank_slosh049 import G, H, pulse_scale, run


LENGTHS = (6.0, 3.0)
ZETAS = (0.0, 0.005, 0.020, 0.050)


def natural_period(length: float) -> float:
    k = np.pi / length
    return float(2 * np.pi / np.sqrt(G * k * np.tanh(k * H)))


T3 = natural_period(3.0)
T6 = natural_period(6.0)
PERIODS = (0.90 * T3, T3, T6, 1.10 * T6)
LABELS = ("0.90T3", "T3", "T6", "1.10T6")


def probe(n_max: int = 31, max_step: float = 0.02, rtol: float = 1e-9):
    rows = []
    for protocol in ("fixed8", "two_cycles"):
        for zeta in ZETAS:
            for label, period in zip(LABELS, PERIODS):
                duration = 8.0 if protocol == "fixed8" else 2 * period
                for length in LENGTHS:
                    result = run(period, duration, zeta, n_max=n_max,
                                 max_step=max_step, rtol=rtol,
                                 tank_length=length)
                    result.pop("solution")
                    rows.append({
                        "protocol": protocol, "zeta": zeta, "label": label,
                        "period_s": period, "duration_s": duration,
                        "length_m": length,
                        "rise_mm": result["during_mm"],
                        "peak_at_s": result["during_at_s"],
                        "n_max": n_max,
                    })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--n-max", type=int, default=31)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps({"T3": T3, "T6": T6, "rows": probe(args.n_max)},
                         indent=2))
