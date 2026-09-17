"""Which fill level makes the water climb highest? Nine fills, one sway.

    python reproduce.py            # every registered control and all 36 cells (a few minutes)
    python reproduce.py --quick    # the controls and the nine undamped cells

WHAT IS CLAIMED

In an ideal two-dimensional linear model of a rigid rectangular tank, 6.00 m long and 3.00 m
tall, filled to nine registered depths from 0.30 m to 2.70 m and swayed by the same horizontal
motion -- a_x(t) = 0.005 g sin(2 pi t / 4 s), for eight seconds -- the 1.20 m fill (40 % full)
produced the largest end-wall water rise of the nine: 77.5 mm. It stayed first in all three
registered damping sensitivities.

WHAT IS NOT

Nothing about a physical vessel. The model is inviscid, linear and two-dimensional, with rigid
walls; the water stays inside it, waves do not break, and there are no transverse modes and no
fluid-structure coupling. The ranking holds only within the eight-second window: the 0.90 m fill
reached its maximum exactly at 8.000 s while still rising, and no motion beyond 8.000 s is
computed. No mechanism for the ranking is claimed.

WHAT THIS DOES

Checks all three modules against the SHA-256 frozen with the reference, runs the registered
controls, re-solves the registered cells, re-applies the registered kill line, and fails if a
number or an outcome has moved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "solver"))
os.environ["TANK_DATA"] = tempfile.mkdtemp(prefix="tanks-")

import tank_slosh as T                  # noqa: E402
import tank_slosh_controls as C         # noqa: E402
import tank_slosh_run as R              # noqa: E402

REF = os.path.join(HERE, "reference", "reference_canonical.json")


def sha(path):
    """SHA-256 over LF-normalised bytes, so the check means the same on a Linux clone, a Windows
    checkout that converts to CRLF, and the tree it was frozen from."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def check_sources(ref):
    bad = [k for k, want in ref["source_sha256"].items()
           if sha(os.path.join(HERE, "solver", k + ".py")) != want]
    if bad:
        raise SystemExit("these do not match the reference: %s" % ", ".join(bad))
    print("   %d modules match the reference by SHA-256" % len(ref["source_sha256"]))


def controls():
    res = C.run(log=lambda m: print("   " + m))
    if not res["passed"]:
        raise SystemExit("a registered control failed")
    return res


def compare(rows, ref, tol):
    largest = 0.0
    for r in rows:
        key = "h%.2f_z%.3f" % (r["h"], r["zeta"])
        want = ref["cells"][key]
        d = abs(r["A_mm"] - want["A_mm"])
        largest = max(largest, d)
        print("   h=%.2f zeta=%.3f  A %8.3f mm   reference %8.3f   difference %.2e"
              % (r["h"], r["zeta"], r["A_mm"], want["A_mm"], d))
    return largest


def quick(ref):
    tol = ref["tolerances"]["A_abs_mm"]
    rows = []
    for h in T.DEPTHS:
        pk = T.Tank(h).peak()
        rows.append(dict(h=h, zeta=0.0, A_mm=pk["A"] * 1000.0, t_peak=pk["t"]))
    largest = compare(rows, ref, tol)
    best = max(rows, key=lambda r: r["A_mm"])
    print("   largest difference %.2e mm (tolerance %.2e); first: %.2f m" % (largest, tol, best["h"]))
    if largest > tol:
        raise SystemExit("the end-wall rises have moved")
    if abs(best["h"] - ref["claimed_h"]) > 1e-9:
        raise SystemExit("the first-ranked fill has moved")


def full(ref):
    code = R.main()
    res = json.load(open(os.path.join(os.environ["TANK_DATA"], "tank_results.json"),
                         encoding="utf-8"))
    rows = list(res["primary"]) + [r for v in res["damped"].values() for r in v]
    largest = compare(rows, ref, ref["tolerances"]["A_abs_mm"])
    for k, want in ref["verdicts"].items():
        got = res["verdicts"][k]
        if got["passed"] != want["passed"] or abs(got["argmax_h"] - want["argmax_h"]) > 1e-9:
            raise SystemExit("the verdict %s has moved: %r against %r" % (k, got, want))
    print("   largest difference %.2e mm across %d cells; all four verdicts unchanged"
          % (largest, len(rows)))
    if code != 0 or largest > ref["tolerances"]["A_abs_mm"]:
        raise SystemExit("a registered number or outcome has moved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ref = json.load(open(REF, encoding="utf-8"))
    print("which-fill-climbs-highest: reproducing the registered study")
    check_sources(ref)
    if a.quick:
        controls()
        quick(ref)
    else:
        full(ref)
    print("OK: every checked number came back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
