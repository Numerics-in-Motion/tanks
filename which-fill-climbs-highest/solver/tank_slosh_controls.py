"""The eight registered controls. All must pass before anything is rendered.

Each one checks the solver against something it cannot influence: an
independently discretised eigenvalue, a closed-form forced response, the static
tilted surface, a symmetry, a conservation law, and its own refinement.

Control 0 is not in the registration and is not reported as evidence: it runs
the whole battery against a deliberately broken solver, because a control that
cannot fail is not a control.
"""
from __future__ import annotations

import numpy as np

from tank_slosh import (A_AMP, DEPTHS, G, L_TANK, N_PROD, N_REFINE, T_END,
                        T_FORCE, Tank)


def c1_dispersion(depths=DEPTHS, n_max=41):
    """omega_n^2 = g k tanh(k h) against an independent integration of the
    vertical problem.

    The velocity potential satisfies phi'' = k^2 phi on the depth with phi'(0) = 0
    at the floor, and the free surface gives omega^2 = g phi'(h) / phi(h).  Rather
    than discretise phi -- which is cosh(k z) and overflows long before the
    tolerance is reachable -- integrate the logarithmic derivative

        psi = phi'/phi,     psi' = k^2 - psi^2,     psi(0) = 0

    which stays bounded in [0, k] and lands on psi(h) = k tanh(k h).  **tanh is
    never called here**, so agreement is a real check on the dispersion relation.
    """
    from scipy.integrate import solve_ivp
    worst = 0.0
    for h in depths:
        t = Tank(h, n_max=n_max)
        for k, w in zip(t.k, t.omega):
            r = solve_ivp(lambda _z, y, kk=k: kk ** 2 - y ** 2, (0.0, h), [0.0],
                          method="DOP853", rtol=1e-13, atol=1e-14)
            w2 = G * float(r.y[0, -1])
            worst = max(worst, abs(w2 - w ** 2) / w ** 2)
    return dict(name="dispersion vs independent vertical integration", worst_rel=worst,
                limit=1e-10, passed=worst < 1e-10)


def c2_exact(depths=DEPTHS):
    """Every undamped modal trajectory against the closed-form forced response."""
    worst_mm, worst_t = 0.0, 0.0
    for h in depths:
        t = Tank(h)
        sol = t.integrate(None)
        ts = np.linspace(0.0, T_END, 20001)
        num = t.wall(sol, ts)
        ex = t.exact_wall(ts)
        worst_mm = max(worst_mm, float(np.max(np.abs(num - ex))) * 1000.0)
        i_n, i_e = int(np.argmax(np.abs(num))), int(np.argmax(np.abs(ex)))
        worst_t = max(worst_t, abs(ts[i_n] - ts[i_e]))
    return dict(name="integrator vs closed-form modal response", worst_mm=worst_mm,
                worst_peak_time_s=worst_t, limits=(0.02, 0.002),
                passed=worst_mm < 0.02 and worst_t < 0.002)


def c3_static(depths=DEPTHS):
    """Under a constant acceleration the surface must be the tilted plane
    eta = -a (x - L/2) / g, with end-to-end difference a L / g."""
    worst_interior, worst_ends, tail = 0.0, 0.0, 0.0
    for h in depths:
        t = Tank(h)
        q = t.force * A_AMP / t.omega ** 2                  # static modal response
        x = np.linspace(0.0, L_TANK, 601)
        eta = (q[:, None] * np.cos(t.k[:, None] * x[None, :])).sum(axis=0)
        exact = -A_AMP * (x - L_TANK / 2) / G
        inner = (x > 0.1 * L_TANK) & (x < 0.9 * L_TANK)
        worst_interior = max(worst_interior, float(np.max(np.abs(eta - exact)[inner])))
        worst_ends = max(worst_ends, abs((eta[0] - eta[-1]) - A_AMP * L_TANK / G))
        tail = max(tail, 4 * A_AMP * L_TANK / (np.pi ** 2 * G * N_PROD))
    return dict(name="static tilted surface under constant acceleration",
                interior_max_m=worst_interior, end_to_end_err_m=worst_ends,
                modal_tail_bound_m=tail,
                passed=worst_interior < 1e-6 and worst_ends < tail)


def c4_zero(depths=DEPTHS):
    """No acceleration, no response."""
    worst = 0.0
    for h in depths:
        t = Tank(h)
        t.force = t.force * 0.0
        sol = t.integrate(None)
        worst = max(worst, float(np.max(np.abs(t.wall(sol, np.linspace(0, T_END, 2001))))))
    return dict(name="zero acceleration gives zero response", worst_m=worst,
                limit=1e-14, passed=worst < 1e-14)


def c5_mirror(depths=DEPTHS):
    """Reversing the sway mirrors the surface and leaves A(h) unchanged."""
    worst = 0.0
    for h in depths:
        a = Tank(h)
        b = Tank(h)
        b.force = -b.force
        pa, pb = a.peak(a.integrate(None)), b.peak(b.integrate(None))
        worst = max(worst, abs(pa["A"] - pb["A"]) * 1000.0)
    return dict(name="sign reversal mirrors the surface", worst_mm=worst,
                limit=1e-9, passed=worst < 1e-9)


def c6_mean(depths=DEPTHS):
    """The spatial mean of eta must stay zero: the water is conserved."""
    worst = 0.0
    for h in depths:
        t = Tank(h)
        sol = t.integrate(None)
        b = t.basis(sol)
        worst = max(worst, b["zero_mean"])
    return dict(name="spatial mean of the surface stays zero", worst_m=worst,
                limit=1e-12, passed=worst < 1e-12)


def c7_energy(depths=DEPTHS):
    """Modal energy minus the work the sway did on it, undamped."""
    from tank_slosh import accel
    worst = 0.0
    for h in depths:
        t = Tank(h)
        sol = t.integrate(None)
        ts = np.linspace(0.0, T_END, 40001)
        m = t.n.size
        y = np.empty((2 * m, ts.size))
        for i in range(0, ts.size, 4096):
            y[:, i:i + 4096] = sol.sol(ts[i:i + 4096])
        q, qd = y[:m], y[m:]
        E = 0.5 * (qd ** 2 + (t.omega ** 2)[:, None] * q ** 2)
        power = (t.force[:, None] * accel(ts)[None, :]) * qd
        W = np.cumsum(0.5 * (power[:, 1:] + power[:, :-1]) * np.diff(ts)[None, :], axis=1)
        err = np.max(np.abs(E[:, 1:].sum(axis=0) - W.sum(axis=0)))
        worst = max(worst, err / max(E.sum(axis=0).max(), 1e-30))
    return dict(name="modal energy equals the work done on it", worst_rel=worst,
                limit=1e-8, passed=worst < 1e-8)


def c8_refinement(depths=DEPTHS):
    """A at 1023 modes against A at 511."""
    worst, rows = 0.0, []
    for h in depths:
        a = Tank(h, n_max=N_PROD).peak()
        b = Tank(h, n_max=N_REFINE).peak()
        d = abs(a["A"] - b["A"]) * 1000.0
        rows.append(dict(h=h, A511_mm=a["A"] * 1000, A1023_mm=b["A"] * 1000, diff_mm=d))
        worst = max(worst, d)
    return dict(name="mode-count refinement", worst_mm=worst, limit=0.01,
                rows=rows, passed=worst < 0.01)


CONTROLS = (c1_dispersion, c2_exact, c3_static, c4_zero, c5_mirror, c6_mean,
            c7_energy, c8_refinement)


def run(which=CONTROLS, log=print):
    out, ok = [], True
    for fn in which:
        r = fn()
        ok = ok and r["passed"]
        log("   %-52s %s" % (r["name"], "PASS" if r["passed"] else "FAIL"))
        for k, v in r.items():
            if k not in ("name", "passed", "rows"):
                log("        %-22s %s" % (k, v))
        out.append(r)
    return dict(passed=ok, controls=out)


if __name__ == "__main__":
    import sys
    print("REGISTERED CONTROLS")
    res = run()
    print("   -> %s" % ("ALL PASS" if res["passed"] else "A CONTROL FAILED"))
    sys.exit(0 if res["passed"] else 1)
