"""Shared matplotlib style for all figures in this project.

Every figure must be readable on both a white and a near-black background
(the figures are embedded transparently in a theme-aware web page), so all
text/spines/grid use a single mid-grey and data uses a small fixed
colorblind-friendly palette.
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

GREY = "#8a8a8a"
BLUE = "#4C9AFF"
ORANGE = "#FF7A59"
GREEN = "#39D98A"
PURPLE = "#C77DFF"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE]

plt.rcParams.update(
    {
        "figure.figsize": (7, 4.2),
        "text.color": GREY,
        "axes.labelcolor": GREY,
        "axes.edgecolor": GREY,
        "axes.titlecolor": GREY,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "grid.color": GREY,
        "grid.alpha": 0.25,
        "legend.labelcolor": GREY,
        "legend.edgecolor": GREY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def new_figure(figsize=(7, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GREY)
    ax.tick_params(colors=GREY)
    return ax


def save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {path}")
    return path
