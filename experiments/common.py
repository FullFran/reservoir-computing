"""Shared experiment configuration (project-wide defaults)."""

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

N = 300
DENSITY = 0.1
T = 6000
WASHOUT = 500
LAM = 1e-6
INPUT_SCALE = 1.0

N_SEEDS = 10
SEEDS = list(range(N_SEEDS))
