"""Exact linear modal response for registered Spinoff 049 divider study.

The smooth finite pulse is a sum of three sinusoids. Each damped modal
oscillator has a closed-form response from rest. The exact-resonance limit is
handled explicitly for the undamped anchor cases.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar

from tank_slosh049 import A_PEAK, G, H, pulse_scale, run
from tank_slosh049_divider import LABELS, LENGTHS, PERIODS, ZETAS, T3, T6


WALL_HEIGHT = 3.0


def modes(length: float, n_max: int):
    n = np.arange(1, n_max + 1, 2, dtype=float)
    k = n * np.pi / length
    depth = np.tanh(k * H)
    w = np.sqrt(G * k * depth)
    coupling = 4 * depth / (n * np.pi)
    return k, w, coupling


def _sine_response(t: np.ndarray, w: np.ndarray, zeta: float,
                   drive: float, strength: np.ndarray) -> np.ndarray:
    """q(t) for q''+2ζωq'+ω²q = strength*sin(drive*t), q(0)=q'(0)=0."""
    tt = t[:, None]
    ww = w[None, :]
    ff = strength[None, :]
    if zeta == 0:
        denom = ww**2 - drive**2
        near = np.abs(denom) < 1e-10 * ww**2
        safe = np.where(near, 1.0, denom)
        out = ff * (np.sin(drive * tt) - drive / ww * np.sin(ww * tt)) / safe
        resonant = ff * (np.sin(ww * tt) - ww * tt * np.cos(ww * tt)) / (2 * ww**2)
        return np.where(near, resonant, out)

    denom = (ww**2 - drive**2)**2 + (2 * zeta * ww * drive)**2
    p0 = -ff * 2 * zeta * ww * drive / denom
    pd0 = ff * drive * (ww**2 - drive**2) / denom
    wd = ww * np.sqrt(1 - zeta**2)
    aa = -p0
    bb = (-pd0 + zeta * ww * aa) / wd
    particular = ff * ((ww**2 - drive**2) * np.sin(drive * tt)
                       - 2 * zeta * ww * drive * np.cos(drive * tt)) / denom
    homogeneous = np.exp(-zeta * ww * tt) * (aa * np.cos(wd * tt)
                                                + bb * np.sin(wd * tt))
    return particular + homogeneous


def modal_response(t, length: float, period: float, duration: float,
                   zeta: float, n_max: int = 511):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any((t < 0) | (t > duration)):
        raise ValueError("registered response is defined during the input only")
    k, w, coupling = modes(length, n_max)
    strength = coupling * pulse_scale(period, duration)
    drive = 2 * np.pi / period
    sideband = 2 * np.pi / duration
    result = np.zeros((len(t), len(w)))
    for coefficient, frequency in ((0.5, drive),
                                   (-0.25, drive + sideband),
                                   (-0.25, drive - sideband)):
        result += coefficient * _sine_response(t, w, zeta, frequency, strength)
    return result


def wall(t, length: float, period: float, duration: float,
         zeta: float, n_max: int = 511):
    return modal_response(t, length, period, duration, zeta, n_max).sum(axis=1)


def peak(length: float, period: float, duration: float, zeta: float,
         n_max: int = 511, scan_step: float = 0.002):
    grid = np.linspace(0, duration, int(np.ceil(duration / scan_step)) + 1)
    samples = np.abs(wall(grid, length, period, duration, zeta, n_max))
    i = int(np.argmax(samples))
    left = float(grid[max(i - 1, 0)])
    right = float(grid[min(i + 1, len(grid) - 1)])
    refined = minimize_scalar(
        lambda t: -abs(wall([t], length, period, duration, zeta, n_max)[0]),
        bounds=(left, right), method="bounded", options={"xatol": 1e-10})
    if -refined.fun > samples[i]:
        return float(-refined.fun), float(refined.x)
    return float(samples[i]), float(grid[i])


def conservative_basis(length: float, period: float, duration: float,
                       zeta: float, n_max: int = 511):
    grid = np.linspace(0, duration, 801)
    q = modal_response(grid, length, period, duration, zeta, n_max)
    k, _, _ = modes(length, n_max)
    max_eta_bound = float(np.max(np.sum(np.abs(q), axis=1)))
    max_slope_bound = float(np.max(np.sum(np.abs(q) * k[None, :], axis=1)))
    return {
        "eta_over_h_upper_bound": max_eta_bound / H,
        "slope_upper_bound": max_slope_bound,
        "freeboard_lower_bound_m": WALL_HEIGHT - H - max_eta_bound,
    }


def table(n_max: int = 511, scan_step: float = 0.002):
    rows = []
    for protocol in ("fixed8", "two_cycles"):
        for zeta in ZETAS:
            for label, period in zip(LABELS, PERIODS):
                duration = 8.0 if protocol == "fixed8" else 2 * period
                for length in LENGTHS:
                    value, when = peak(length, period, duration, zeta,
                                       n_max, scan_step)
                    row = dict(protocol=protocol, zeta=zeta, label=label,
                               period_s=period, duration_s=duration,
                               length_m=length, rise_mm=1000 * value,
                               peak_at_s=when, n_max=n_max)
                    row.update(conservative_basis(length, period, duration,
                                                  zeta, n_max))
                    rows.append(row)
    return rows


def ode_control(length: float, period: float, duration: float,
                zeta: float, n_max: int = 31):
    """Independent DOP853 integration compared with exact mode sum at a grid."""
    integration = run(period, duration, zeta, n_max=n_max,
                      tank_length=length, max_step=0.005, rtol=1e-11)
    solution = integration.pop("solution")
    grid = np.linspace(0, duration, 1001)
    numeric = solution.sol(grid)[:len(modes(length, n_max)[0])].sum(axis=0)
    exact = wall(grid, length, period, duration, zeta, n_max)
    return float(np.max(np.abs(numeric - exact)) * 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--n-max", type=int, default=511)
    args = parser.parse_args()
    if args.table:
        print(json.dumps({"T3": T3, "T6": T6,
                          "rows": table(args.n_max)}, indent=2))
