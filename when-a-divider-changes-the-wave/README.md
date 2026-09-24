# When a divider changes the wave

This is a numerical result for an **ideal two-dimensional linear tank**, not an experiment or vessel-design recommendation. A rigid 6.00 m tank holds water 1.20 m deep. A sealed, impermeable, full-height central divider makes two independent 3.00 m compartments. The exterior footprint and total water depth are unchanged. Each configuration receives the same prescribed horizontal acceleration with an actual peak of 0.001 g: a sinusoid under a smooth `sin²` envelope, for 8.00 s.

At a 2.12648 s forcing period (the 3 m compartment's first-mode period), the maximum absolute end-wall rise **during the eight-second input** changes from 2.057 mm (undivided) to 7.247 mm (divided). At 3.71565 s (the 6 m tank's first-mode period), it changes from 8.710 mm to 2.965 mm. Thus a central divider amplifies one period while suppressing the other. Each maximum is taken independently within 0–8 s; the two tanks need not peak at the same instant. Neighboring periods and damping sensitivities preserve these directions. A separate equal-two-cycles duration check preserves direction but not the quoted gap.

The divider is sealed to the floor and side walls and reaches the open rim; **it is not a porous or partial baffle**. The model assumes small, nonbreaking waves; rigid boundaries; inviscid, incompressible, irrotational flow; a linear free surface; and no transverse mode or fluid–structure coupling. The predicted millimetres and ranking are conditional on those assumptions, the input amplitude and envelope, and the eight-second window. No post-input response or physical performance is claimed.

Odd-mode surface elevation is `eta(x,t) = sum q_n(t) cos(n pi x/L)`, with `omega_n² = g (n pi/L) tanh(n pi h/L)` and `q_n'' + 2 zeta omega_n q_n' + omega_n² q_n = [4 tanh(n pi h/L)/(n pi)] a_x(t)`. The smooth drive is a sum of three sinusoids, so each modal response is evaluated analytically from rest; the exact-resonance limit is handled explicitly. All rows use odd modes through `n=511`. The solution was checked against independent DOP853 integration, `n=1023` refinement, and peak-scan refinement. Across all 64 registered rows, conservative bounds remain below 0.007874 for `|eta|/h` and 0.007851 for surface slope, with at least 1.7905 m of freeboard.

The video enlarges **only the vertical departure from still water** by 25×. Its millimetre labels are the true model values, not the enlarged picture. The short photographic illustration is conceptual; the animated water fields come from the model.

Run from this directory after installing `requirements.txt`:

```
python reproduce.py --quick
python reproduce.py
```

The frozen canonical table is `reference/reference_canonical.json`. The reproduction checks the source hashes, an independent ODE control, every selected numerical cell and the sign of the stated comparison. The full run checks all 64 registered cells; `--quick` checks the eight primary cells. The reference was exported from the production calculation, not recomputed by this package.
