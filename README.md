# tanks

Water in rigid tanks, solved from the equations and published with everything needed to run it
again.

| study | question |
|---|---|
| [`which-fill-climbs-highest/`](which-fill-climbs-highest/) | Nine fill levels, one horizontal sway: which one sends the water highest up the end wall? |

Each study folder holds its solver, a frozen reference of every reported number with the SHA-256
of the modules that produced it, and a `reproduce.py` that re-runs the controls and the
registered cells and fails if anything has moved.

```
pip install -r requirements.txt
cd which-fill-climbs-highest
python reproduce.py --quick
```
