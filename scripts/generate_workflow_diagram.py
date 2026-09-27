"""Draw the methodology workflow diagram used as Figure 2.1 of the thesis.

The diagram replaces an earlier hand-written SVG that was laid out in wide rows.
That layout had to be scaled to about a third of its natural size to fit the text
block, which made the labels unreadable in print. This version is portrait and is
drawn at close to its printed size, so the point sizes below are what the reader
sees on the page.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from loguru import logger

RUN = os.environ.get("THESIS_RUN_VERSION", "v6")
FIGDIR = Path(os.environ.get(
    "THESIS_FIGURE_DIR", Path("past-runs") / RUN / "outputs" / "figures"))
FIGDIR.mkdir(parents=True, exist_ok=True)

FIG_W = 6.0          # inches; the thesis text block is 5.42 in
UNITS_X = 100.0      # drawing units across the full figure width

BOX_W = 45.0
BOX_H = 8.6
COL_X = (3.0, 52.0)
ROW_GAP = 2.2
BAND_PAD = 1.8
BAND_TITLE_H = 5.4
SECTION_GAP = 2.8

HEAD_SIZE = 10.0
SUB_SIZE = 8.2
BAND_SIZE = 10.5
ICON_SIZE = 7.5

SECTIONS = [
    {
        "title": "1. Data intake, quality control and features",
        "band": "#dbeafe",
        "edge": "#93c5fd",
        "accent": "#2563eb",
        "boxes": [
            ("DB", "Raw streams", "ActMindata, X/Y/Z means, time"),
            ("QC", "Deployment QC", "documented removal windows"),
            ("FX", "Feature build", "log, rolling, per-subject z-score"),
            ("TM", "Time tags", "hour of day"),
        ],
    },
    {
        "title": "2\u20133. Behavioural state inference and validation",
        "band": "#dcfce7",
        "edge": "#86efac",
        "accent": "#16a34a",
        "boxes": [
            ("HM", "HMM fit and decode", "per subject, $K$ = 2\u20135, BIC, Viterbi"),
            ("DW", "Dwell filter", "30 min; 15/30/60 sensitivity"),
            ("PV", "Posture validation", "independent check of ordering"),
            ("HS", "HSMM extension", "Poisson state durations"),
        ],
    },
    {
        "title": "4. Multi-component cosinor rhythmometry",
        "band": "#ffedd5",
        "edge": "#fdba74",
        "accent": "#ea580c",
        "boxes": [
            ("C1", "Multi-cosinor model", "24, 12, 8, 6, 4.8 h harmonics"),
            ("RF", "Rhythm fit", "NLS; amplitudes and phases"),
            ("EV", "Rhythm testing", "$R^2$ and F-test"),
        ],
    },
    {
        "title": "5. Covariate effects (LMM) and outputs",
        "band": "#ede9fe",
        "edge": "#c4b5fd",
        "accent": "#7c3aed",
        "boxes": [
            ("NP", "Exploratory tests", "Mann-Whitney, Kruskal-Wallis"),
            ("LM", "Cosinor mixed models", "sex; random slopes; ICC, $R^2$"),
            ("CP", "HMM vs HSMM", "duration comparison"),
            ("OUT", "Outputs", "figures, tables"),
        ],
    },
]

# The figure title and the caveat note live in the LaTeX caption, not in the
# artwork, so the drawing area is spent entirely on legible box labels.
HEADER_H = 1.8
FOOTER_H = 1.8


def _section_height(n_boxes: int) -> float:
    rows = (n_boxes + 1) // 2
    return BAND_TITLE_H + rows * BOX_H + (rows - 1) * ROW_GAP + 2 * BAND_PAD


def _total_height() -> float:
    body = sum(_section_height(len(s["boxes"])) for s in SECTIONS)
    body += SECTION_GAP * (len(SECTIONS) - 1)
    return HEADER_H + body + FOOTER_H


def _box(ax, x, y, icon, head, sub, accent, edge):
    """Draw one rounded step box with its circular badge; (x, y) is the top-left."""
    ax.add_patch(FancyBboxPatch(
        (x, y - BOX_H), BOX_W, BOX_H,
        boxstyle="round,pad=0,rounding_size=1.2",
        facecolor="white", edgecolor=edge, linewidth=1.1, zorder=2))

    cx, cy = x + 4.9, y - BOX_H / 2
    ax.add_patch(plt.Circle((cx, cy), 3.2, facecolor=accent, edgecolor="none",
                            zorder=3))
    ax.text(cx, cy, icon, ha="center", va="center", color="white",
            fontsize=ICON_SIZE, fontweight="bold", zorder=4)

    tx = x + 9.8
    ax.text(tx, y - BOX_H / 2 + 1.8, head, ha="left", va="center",
            fontsize=HEAD_SIZE, fontweight="bold", color="#0f172a", zorder=4)
    ax.text(tx, y - BOX_H / 2 - 1.9, sub, ha="left", va="center",
            fontsize=SUB_SIZE, color="#334155", zorder=4)


def _arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8,
        color="#334155", linewidth=1.1, shrinkA=0, shrinkB=0, zorder=5))


def build() -> Path:
    total_h = _total_height()
    fig_h = FIG_W * total_h / UNITS_X
    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, UNITS_X)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (0.4, 0.4), UNITS_X - 0.8, total_h - 0.8,
        boxstyle="round,pad=0,rounding_size=1.5",
        facecolor="#f6faff", edgecolor="#dbe6f5", linewidth=1.0, zorder=0))

    y = total_h - HEADER_H
    for si, section in enumerate(SECTIONS):
        boxes = section["boxes"]
        height = _section_height(len(boxes))

        ax.add_patch(FancyBboxPatch(
            (2.0, y - height), UNITS_X - 4.0, height,
            boxstyle="round,pad=0,rounding_size=1.5",
            facecolor=section["band"], edgecolor="none", alpha=0.75, zorder=1))
        ax.text(3.6, y - BAND_PAD - 0.4, section["title"], ha="left", va="top",
                fontsize=BAND_SIZE, fontweight="bold", color="#0f172a", zorder=4)

        top = y - BAND_PAD - BAND_TITLE_H
        corners = []
        for bi, (icon, head, sub) in enumerate(boxes):
            row = bi // 2
            # Serpentine order, so consecutive steps are always orthogonal neighbours.
            col = bi % 2 if row % 2 == 0 else 1 - (bi % 2)
            bx = COL_X[col]
            by = top - row * (BOX_H + ROW_GAP)
            _box(ax, bx, by, icon, head, sub, section["accent"], section["edge"])
            corners.append((bx, by))

        for bi in range(len(boxes) - 1):
            (x0, y0), (x1, y1) = corners[bi], corners[bi + 1]
            mid0, mid1 = y0 - BOX_H / 2, y1 - BOX_H / 2
            if y0 == y1:
                if x1 > x0:
                    _arrow(ax, (x0 + BOX_W, mid0), (x1, mid1))
                else:
                    _arrow(ax, (x0, mid0), (x1 + BOX_W, mid1))
            else:
                _arrow(ax, (x0 + BOX_W / 2, y0 - BOX_H), (x1 + BOX_W / 2, y1))

        if si < len(SECTIONS) - 1:
            _arrow(ax, (UNITS_X / 2, y - height),
                   (UNITS_X / 2, y - height - SECTION_GAP + 0.3))
        y -= height + SECTION_GAP

    path = FIGDIR / "methodology_workflow_diagram.png"
    fig.savefig(path, dpi=400)
    plt.close(fig)
    logger.success(f"wrote {path} ({FIG_W:.2f} x {fig_h:.2f} in)")
    return path


if __name__ == "__main__":
    build()
