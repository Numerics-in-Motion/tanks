"""The registered production run: controls first, then every registered cell.

One batch into an empty directory. Nothing here chooses anything -- the depths,
the damping set, the mode counts, the tolerances, the uncertainty sum and the
kill line were all frozen before the first cell was computed.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import tank_slosh_controls as C
from tank_slosh import (CLAIMED, DATA, DEPTHS, ETA_OVER_H, FREEBOARD, N_PROD,
                        N_REFINE, REFINE_LIMIT, ROUND_FLOOR, SLOPE_LIMIT, T_END,
                        U_LIMIT, ZETAS, Tank)


def _json(o):
    if isinstance(o, dict):
        return {k: _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    return o


def cell(h, zeta, log=print):
    """One registered cell: production, refinement, the closed form, and the basis."""
    t0 = time.time()
    prod = Tank(h, zeta=zeta, n_max=N_PROD)
    sol = prod.integrate(None)
    pk = prod.peak(sol)

    fine = Tank(h, zeta=zeta, n_max=N_REFINE)
    pk_f = fine.peak()

    # the independent closed form, at the production mode count
    ts = np.linspace(0.0, T_END, 80001)
    ex = prod.exact_wall(ts)
    i = int(np.argmax(np.abs(ex)))
    d_ode = abs(abs(ex[i]) - pk["A"]) * 1000.0
    d_peak = abs(pk["A"] - pk["A_grid"]) * 1000.0
    d_ref = abs(pk["A"] - pk_f["A"]) * 1000.0

    U = 2 * d_ref + d_ode + d_peak + ROUND_FLOOR
    b = prod.basis(sol)
    gates = dict(
        eta_over_h=(b["eta_over_h"], ETA_OVER_H, b["eta_over_h"] <= ETA_OVER_H),
        slope=(b["slope"], SLOPE_LIMIT, b["slope"] <= SLOPE_LIMIT),
        freeboard=(b["freeboard"], FREEBOARD, b["freeboard"] >= FREEBOARD),
        positive_depth=(b["min_depth"], 0.0, b["min_depth"] > 0.0),
        refinement=(d_ref, REFINE_LIMIT, d_ref <= REFINE_LIMIT),
        uncertainty=(U, U_LIMIT, U <= U_LIMIT),
    )
    ok = all(v[2] for v in gates.values())
    log("   h=%.2f zeta=%.3f  A=%8.3f mm  t*=%6.3f s  U=%.3f mm  %s  (%.1f s)"
        % (h, zeta, pk["A"] * 1000, pk["t"], U, "ok" if ok else "GATE FAIL", time.time() - t0))
    return dict(h=h, zeta=zeta, A_mm=pk["A"] * 1000, t_peak=pk["t"],
                A_refined_mm=pk_f["A"] * 1000, d_refine_mm=d_ref, d_ode_mm=d_ode,
                d_peak_mm=d_peak, U_mm=U, T1=b["T1"], basis=b, gates=gates, valid=ok)


def claim(rows):
    """The registered claim and its kill line, applied to one damping level."""
    best = max(rows, key=lambda r: r["A_mm"])
    win = next(r for r in rows if abs(r["h"] - CLAIMED) < 1e-9)
    others = [r for r in rows if r is not win]
    seps = [dict(h=r["h"], sep=win["A_mm"] - r["A_mm"],
                 need=max(1.0, 3 * (win["U_mm"] + r["U_mm"])),
                 ok=(win["A_mm"] - r["A_mm"]) > max(1.0, 3 * (win["U_mm"] + r["U_mm"])))
            for r in others]
    ends = max(r["A_mm"] for r in rows if abs(r["h"] - 0.30) < 1e-9 or abs(r["h"] - 2.70) < 1e-9)
    return dict(argmax_h=best["h"], claimed_h=CLAIMED,
                unique_winner=abs(best["h"] - CLAIMED) < 1e-9,
                A_claimed_mm=win["A_mm"], separations=seps,
                all_separated=all(s["ok"] for s in seps),
                end_ratio=win["A_mm"] / ends, end_ratio_ok=win["A_mm"] > 1.25 * ends,
                passed=(abs(best["h"] - CLAIMED) < 1e-9 and all(s["ok"] for s in seps)
                        and win["A_mm"] > 1.25 * ends))


def main():
    os.makedirs(DATA, exist_ok=True)
    log = print
    log("REGISTERED CONTROLS")
    ctl = C.run(log=log)
    if not ctl["passed"]:
        json.dump(_json(dict(controls=ctl, release_blocked=True)),
                  open(os.path.join(DATA, "tank_results.json"), "w"), indent=1)
        log("   -> A REGISTERED CONTROL FAILED. Production not run.")
        return 1
    log("   -> all controls pass")

    log("\nPRIMARY CELLS (undamped)")
    primary = [cell(h, 0.0, log) for h in DEPTHS]
    log("\nDAMPING SENSITIVITIES")
    damped = {z: [cell(h, z, log) for h in DEPTHS] for z in ZETAS}

    verdicts = dict(undamped=claim(primary))
    for z, rows in damped.items():
        verdicts["zeta_%.3f" % z] = claim(rows)

    all_valid = all(r["valid"] for r in primary) and all(
        r["valid"] for rows in damped.values() for r in rows)
    passed = all_valid and all(v["passed"] for v in verdicts.values())

    out = dict(registration="042 tank slosh", controls=ctl, primary=primary,
               damped={("%.3f" % z): rows for z, rows in damped.items()},
               verdicts=verdicts, all_valid=all_valid, passed=passed,
               release_blocked=not passed)
    json.dump(_json(out), open(os.path.join(DATA, "tank_results.json"), "w"), indent=1)

    log("\nRESULT")
    for name, v in verdicts.items():
        log("   %-12s argmax h=%.2f (registered %.2f)  unique=%s  separated=%s  end-ratio %.3f  -> %s"
            % (name, v["argmax_h"], v["claimed_h"], v["unique_winner"],
               v["all_separated"], v["end_ratio"], "PASS" if v["passed"] else "KILLED"))
    log("   every cell valid: %s" % all_valid)
    log("   -> %s" % ("CLAIM STANDS" if passed else "KILL LINE FIRED"))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
