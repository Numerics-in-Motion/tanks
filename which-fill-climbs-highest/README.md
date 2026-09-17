# Which fill level makes the water climb highest?

One rigid rectangular tank, 6.00 m long and 3.00 m tall, filled to nine depths from 10 % to 90 %
full. Every fill receives exactly the same horizontal sway — `a_x(t) = 0.005 g sin(2 pi t / 4 s)`,
for eight seconds and no longer — and the measured quantity is the largest rise of the water at
an end wall during those eight seconds.

## The result

| fill depth (m) | 0.30 | 0.60 | 0.90 | **1.20** | 1.50 | 1.80 | 2.10 | 2.40 | 2.70 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| end-wall rise, 0–8 s (mm) | 13.471 | 38.842 | 70.286 | **77.530** | 72.226 | 64.592 | 57.512 | 51.630 | 48.104 |
| fundamental period (s) | 7.025 | 5.027 | 4.184 | **3.716** | 3.424 | 3.231 | 3.099 | 3.007 | 2.942 |
| time of maximum (s) | 5.216 | 6.674 | 8.000 | **7.680** | 7.361 | 7.131 | 6.972 | 6.862 | 5.069 |

The **1.20 m fill (40 % full) produced the largest rise, 77.5 mm**. Second was 1.50 m at 72.2 mm
and third was 0.90 m at 70.3 mm. The 1.20 m fill stayed first in all three registered damping
sensitivities (damping ratio 0.005, 0.020 and 0.050 on every mode).

The fill with the fundamental period closest to the four-second sway is 0.90 m (4.184 s), and it
did not rank first within the eight-second window.

**The 0.90 m fill reached its maximum exactly at 8.000 s, while its end-wall elevation was still
increasing.** Its 70.3 mm is therefore a maximum within the specified 0–8 s window, and this
ranking covers only that window. No motion beyond 8.000 s is computed, and no mechanism for the
ranking is claimed.

## What is claimed, and what is not

Claimed: under the registered eight-second 0.005 g sway, the 1.20 m fill produced the largest
end-wall water rise among the nine tested depths in this ideal linear tank model.

Not claimed: anything about a physical vessel. This is a two-dimensional unit-width slice with
rigid walls and floor; the water stays inside it and its waves do not break. The fluid is
inviscid, and there is no transverse mode, no wall flexibility and no fluid-structure coupling. Nothing here
is a structural, operating or design figure, and 1.20 m is the highest of nine sampled depths —
nothing between them is computed.

## The model

Inviscid, incompressible, irrotational, linear free surface. The surface elevation is the sum of
the odd cosine modes of the tank:

```
eta(x, t) = sum_{n odd} q_n(t) cos(k_n x),        k_n = n pi / L
omega_n^2 = g k_n tanh(k_n h)
q_n'' + 2 zeta omega_n q_n' + omega_n^2 q_n = 4 a_x(t) tanh(k_n h) / (n pi)
```

from rest, integrated with DOP853 (relative tolerance 1e-11, absolute 1e-13, maximum step
0.005 s) on odd modes up to n = 511, and refined against n = 1023. Only odd modes are excited by a
horizontal acceleration, so `eta(L, t) = -eta(0, t)` and the rise at either end wall is
`|eta(0, t)|`.

## Checks

`reproduce.py` runs all eight registered controls before any cell:

- the dispersion relation against an independent integration of the vertical problem that does
  not call `tanh` (largest relative error 9.8e-14);
- every undamped modal trajectory against its closed-form forced response (2.4e-13 mm);
- a constant acceleration must give the static tilted surface `eta = -a (x - L/2) / g`;
- zero acceleration must give zero response, and reversing the sway must mirror the surface;
- the spatial mean of `eta` must stay zero, so no water is created or lost (1.4e-17 m);
- modal energy must equal the work the sway did on it (5.9e-9 relative);
- the rise at 1023 modes must agree with 511 modes to 0.01 mm (largest difference 0.0059 mm).

Each cell also carries an uncertainty `U = 2|A_1023 - A_511| + closed-form disagreement +
peak-location refinement + 0.05 mm`, required to be at most 0.10 mm (largest 0.062 mm), and basis
limits on surface height, slope and clearance below the rim (tightest: rise/depth 0.078 against
0.10, slope 0.041 against 0.10, clearance 0.252 m against 0.20 m).

The frozen `reference/reference_canonical.json` holds the SHA-256 of all three modules together
with every reported number.

```
python reproduce.py --quick   # controls and the nine undamped cells
python reproduce.py           # all 36 cells and every verdict
```
