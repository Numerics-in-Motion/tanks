"""Which fill level makes a water tank slosh highest? -- the solver.

REGISTRATION (frozen by the design review before any cell was computed).
Nothing in this file may be changed to suit a result.

THE QUESTION

One rigid rectangular tank, 6.00 m long and 3.00 m tall, is swayed by the same
small horizontal motion at nine fill depths from 10 % to 90 % full. The motion
is fixed: a_x(t) = 0.005 g sin(2 pi t / 4 s), for eight seconds and no longer.
For each depth, how high does the water climb the end wall while that motion
lasts, and which depth climbs highest?

THE MODEL

A two-dimensional unit-width slice. Inviscid, incompressible, irrotational,
linear free surface; rigid walls and floor; water initially level and at rest.
The surface elevation is carried by the odd cosine modes,

    eta(x, t) = sum_{n odd} q_n(t) cos(k_n x),      k_n = n pi / L
    omega_n^2 = g k_n tanh(k_n h)
    q_n'' + 2 zeta omega_n q_n' + omega_n^2 q_n = 4 a_x(t) tanh(k_n h) / (n pi)

with zero initial conditions. Only odd n are driven: the even modes are
symmetric and a horizontal acceleration cannot excite them.

WHAT IS MEASURED

    A(h) = max over 0 <= t <= 8 s of |eta(0, t)|

the largest rise of the water at an end wall during the motion. Because only
odd modes appear, eta(L, t) = -eta(0, t), so one wall's rise is the other's
fall and the maximum over both walls is |eta(0, t)|.

WHAT IS NOT MEASURED, AND NOT CLAIMED

Nothing about a real tank. No safety, overflow, structural force or operating
statement. No wall flexibility, no viscosity, no breaking or overturning waves,
no transverse modes, no fluid-structure coupling, and nothing after the eight
seconds of prescribed motion.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------- registration
L_TANK = 6.00            # m, tank length
H_WALL = 3.00            # m, wall height
RHO = 1000.0             # kg/m^3
G = 9.80665              # m/s^2

A_AMP = 0.005 * G        # m/s^2, sway amplitude
T_FORCE = 4.00           # s, sway period
T_END = 8.00             # s, the motion stops here and the window ends here

DEPTHS = (0.30, 0.60, 0.90, 1.20, 1.50, 1.80, 2.10, 2.40, 2.70)
ZETAS = (0.005, 0.020, 0.050)
CLAIMED = 1.20           # the registered winner; the kill line tests it

N_PROD = 511             # odd modes up to here in production
N_REFINE = 1023          # odd modes up to here in the refinement

RTOL, ATOL, MAX_STEP = 1e-11, 1e-13, 0.005
PEAK_TOL = 1e-5          # s, bracketed refinement of the time maximum
U_LIMIT = 0.10           # mm, required uncertainty on every claimed cell
REFINE_LIMIT = 0.01      # mm, required |A_1023 - A_511|
ROUND_FLOOR = 0.05       # mm, the registered reporting floor

ETA_OVER_H = 0.10        # basis: max |eta| / h
SLOPE_LIMIT = 0.10       # basis: max |d eta / dx|
FREEBOARD = 0.20         # m, required clear wall above the crest

DATA = os.environ.get("TANK_DATA") or os.path.abspath("tank_data")


def accel(t):
    """The registered sway. Zero outside the eight seconds."""
    t = np.asarray(t, dtype=float)
    return np.where((t >= 0.0) & (t <= T_END), A_AMP * np.sin(2 * np.pi * t / T_FORCE), 0.0)


class Tank:
    """One fill depth of the registered tank."""

    def __init__(self, h, zeta=0.0, n_max=N_PROD, L=L_TANK):
        self.h, self.zeta, self.L = h, zeta, L
        self.n = np.arange(1, n_max + 1, 2, dtype=float)      # odd modes only
        self.k = self.n * np.pi / L
        self.tanh = np.tanh(self.k * h)
        self.omega = np.sqrt(G * self.k * self.tanh)
        self.force = 4.0 * self.tanh / (self.n * np.pi)       # multiplies a_x(t)

    # ------------------------------------------------------------- the solve
    def _rhs(self, t, y):
        m = self.n.size
        q, qd = y[:m], y[m:]
        return np.concatenate([qd, self.force * accel(t)
                               - 2 * self.zeta * self.omega * qd - self.omega ** 2 * q])

    def integrate(self, t_eval):
        m = self.n.size
        sol = solve_ivp(self._rhs, (0.0, T_END), np.zeros(2 * m), method="DOP853",
                        t_eval=t_eval, rtol=RTOL, atol=ATOL, max_step=MAX_STEP,
                        dense_output=True)
        if not sol.success:
            raise RuntimeError("integration failed for h=%.2f: %s" % (self.h, sol.message))
        return sol

    # --------------------------------------------------- the independent form
    def exact_q(self, t):
        """Closed-form modal response to the registered sine from rest.

        Independent of the integrator: this is the control, not the product.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        w, z, F = self.omega, self.zeta, self.force * A_AMP
        Om = 2 * np.pi / T_FORCE
        d = (w ** 2 - Om ** 2) ** 2 + (2 * z * w * Om) ** 2
        # particular: F[(w^2-Om^2) sin(Om t) - 2 z w Om cos(Om t)] / d
        p0 = -F * 2 * z * w * Om / d
        pd0 = F * Om * (w ** 2 - Om ** 2) / d
        wd = w * np.sqrt(max(1.0 - z ** 2, 0.0)) if z < 1 else w
        A = -p0
        B = (-pd0 + z * w * A) / wd
        tt = t[:, None]
        part = F[None, :] * ((w ** 2 - Om ** 2)[None, :] * np.sin(Om * tt)
                             - (2 * z * w * Om)[None, :] * np.cos(Om * tt)) / d[None, :]
        hom = np.exp(-z * w[None, :] * tt) * (A[None, :] * np.cos(wd[None, :] * tt)
                                              + B[None, :] * np.sin(wd[None, :] * tt))
        return part + hom

    def exact_wall(self, t):
        return self.exact_q(t).sum(axis=1)

    # --------------------------------------------------------- the estimand
    def wall(self, sol, t, chunk=4096):
        """eta(0, t) = sum q_n(t), since cos(0) = 1 for every mode.

        Evaluated in chunks: the dense output carries 2 x 256 states, and asking
        for the whole scan at once would build a gigabyte-scale array.
        """
        m = self.n.size
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty(t.size)
        for i in range(0, t.size, chunk):
            out[i:i + chunk] = sol.sol(t[i:i + chunk])[:m].sum(axis=0)
        return out

    def peak(self, sol=None):
        """A(h): the largest |eta(0,t)| in the window, refined off a dense scan."""
        sol = sol if sol is not None else self.integrate(None)
        grid = np.linspace(0.0, T_END, 80001)          # 100 us; mode 1 has a ~3-7 s period
        f = np.abs(self.wall(sol, grid))
        i = int(np.argmax(f))
        lo = grid[max(i - 1, 0)]
        hi = grid[min(i + 1, grid.size - 1)]
        r = minimize_scalar(lambda tt: -abs(self.wall(sol, np.array([tt]))[0]),
                            bounds=(lo, hi), method="bounded",
                            options=dict(xatol=PEAK_TOL))
        t_star = float(r.x) if -r.fun >= f[i] else float(grid[i])
        return dict(A=float(abs(self.wall(sol, np.array([t_star]))[0])),
                    t=t_star, A_grid=float(f[i]), t_grid=float(grid[i]))

    # ------------------------------------------------------------ diagnostics
    def surface(self, sol, t, nx=601):
        x = np.linspace(0.0, self.L, nx)
        m = self.n.size
        q = sol.sol(t)[:m]
        return x, (q[:, None] * np.cos(self.k[:, None] * x[None, :])).sum(axis=0)

    def basis(self, sol):
        """Everything the registered basis gates read."""
        ts = np.linspace(0.0, T_END, 2001)
        eta_max = slope_max = 0.0
        mean_worst = 0.0
        for t in ts:
            x, eta = self.surface(sol, t)
            eta_max = max(eta_max, float(np.max(np.abs(eta))))
            slope_max = max(slope_max, float(np.max(np.abs(np.gradient(eta, x)))))
            mean_worst = max(mean_worst, abs(float(np.trapezoid(eta, x) / self.L)))
        crest = eta_max
        return dict(eta_over_h=eta_max / self.h, slope=slope_max,
                    freeboard=H_WALL - self.h - crest, min_depth=self.h - eta_max,
                    zero_mean=mean_worst, eta_max=eta_max,
                    T1=float(2 * np.pi / self.omega[0]))
