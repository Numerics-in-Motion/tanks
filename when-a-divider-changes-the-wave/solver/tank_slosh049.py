"""Smooth drive and independent DOP853 control for Spinoff 049.

The older post-input ``probe`` at the bottom preserves the screened-out
candidate's audit trail; the final divider study uses only ``pulse_scale``
and ``run`` from this module. It does not modify Spinoff 042's solver.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


G = 9.80665
L = 6.0
H = 1.2
A_PEAK = 0.001 * G
PERIODS = (2.5, 3.0, 3.5, 4.0)
ZETAS = (0.005, 0.020, 0.050)
POST_SECONDS = 8.0


def pulse_scale(period: float, duration: float) -> float:
    """Scale the smooth pulse to one exact peak of prescribed acceleration."""
    omega = 2 * np.pi / period

    def shape(t):
        return np.sin(omega * t) * np.sin(np.pi * t / duration) ** 2

    grid = np.linspace(0, duration, 2001)
    values = np.abs(shape(grid))
    candidates = np.argpartition(values, -8)[-8:]
    peak = float(values.max())
    for i in candidates:
        left = float(grid[max(0, i - 1)])
        right = float(grid[min(len(grid) - 1, i + 1)])
        refined = minimize_scalar(lambda t: -abs(shape(t)), bounds=(left, right),
                                  method="bounded", options={"xatol": 1e-13})
        peak = max(peak, float(-refined.fun))
    return A_PEAK / peak


def run(period: float, duration: float, zeta: float, n_max: int = 31,
        max_step: float = 0.02, rtol: float = 1e-9,
        tank_length: float = L) -> dict:
    n = np.arange(1, n_max + 1, 2, dtype=float)
    k = n * np.pi / tank_length
    depth = np.tanh(k * H)
    natural = np.sqrt(G * k * depth)
    coupling = 4 * depth / (n * np.pi)
    scale = pulse_scale(period, duration)
    drive_omega = 2 * np.pi / period
    modal_count = len(n)

    def acceleration(t):
        if t < 0 or t > duration:
            return 0.0
        return scale * np.sin(drive_omega * t) * np.sin(np.pi * t / duration) ** 2

    def rhs(t, state):
        q, qd = state[:modal_count], state[modal_count:]
        return np.concatenate((qd, coupling * acceleration(t)
                               - 2 * zeta * natural * qd - natural**2 * q))

    end = duration + POST_SECONDS
    solution = solve_ivp(rhs, (0.0, end), np.zeros(2 * modal_count),
                         method="DOP853", dense_output=True,
                         rtol=rtol, atol=rtol * 0.01, max_step=max_step)
    if not solution.success:
        raise RuntimeError(solution.message)

    def wall(t):
        state = solution.sol(t)
        return state[:modal_count].sum(axis=0)

    def peak(start, stop):
        grid = np.linspace(start, stop, int(round((stop - start) / 0.002)) + 1)
        samples = np.abs(wall(grid))
        i = int(np.argmax(samples))
        left = float(grid[max(i - 1, 0)])
        right = float(grid[min(i + 1, len(grid) - 1)])
        refined = minimize_scalar(lambda t: -abs(wall(t)), bounds=(left, right),
                                  method="bounded", options={"xatol": 1e-9})
        if -refined.fun > samples[i]:
            return float(-refined.fun), float(refined.x)
        return float(samples[i]), float(grid[i])

    during, during_at = peak(0.0, duration)
    after, after_at = peak(duration, end)
    return {
        "period_s": period, "duration_s": duration, "zeta": zeta,
        "n_max": n_max, "accel_peak_g": A_PEAK / G,
        "tank_length_m": tank_length,
        "during_mm": during * 1000, "during_at_s": during_at,
        "post_mm": after * 1000, "post_at_s": after_at,
        "post_over_during": after / during if during else None,
        "T1_s": float(2 * np.pi / natural[0]),
        "solution": solution,
    }


def probe(n_max: int):
    rows = []
    for protocol in ("fixed8", "two_cycles"):
        for zeta in ZETAS:
            for period in PERIODS:
                duration = 8.0 if protocol == "fixed8" else 2 * period
                result = run(period, duration, zeta, n_max=n_max)
                result.pop("solution")
                result["protocol"] = protocol
                rows.append(result)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--n-max", type=int, default=31)
    arguments = parser.parse_args()
    if arguments.probe:
        print(json.dumps(probe(arguments.n_max), indent=2))
