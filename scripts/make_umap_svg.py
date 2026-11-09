"""Generate an animated SVG of cells resolving from noise into labeled clusters.

Two input modes:
  * synthetic (default): hand-placed kidney cell types, no data needed.
  * --embedding CSV: a real 2-D embedding with columns ``x,y,label``
    (e.g. exported from AnnData by ``export_pbmc_embedding.py``).

Animation is SMIL only, so it plays inside a GitHub README where JavaScript is
stripped. Each circle also carries static ``cx``/``cy`` at its cluster
position, so renderers without SMIL support show the final clustered state
instead of piling every cell at the origin.

Usage:
    python make_umap_svg.py --out assets/umap-dark.svg --theme dark
    python make_umap_svg.py --embedding data/pbmc.csv --out assets/umap-dark.svg \
        --theme dark --caption "pbmc3k · 800 cells · UMAP"
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

WIDTH: int = 820
HEIGHT: int = 420
DURATION_S: float = 9.0

# Plot box keeps clear bands for labels at the top and the caption at the
# bottom; without this the caption is drawn over cells.
PAD_X: float = 24.0
PAD_TOP: float = 34.0
PAD_BOTTOM: float = 40.0

MAX_LABELS: int = 10
FONT: str = "ui-monospace,SFMono-Regular,Menlo,monospace"

# Tableau 10: colorblind-tolerant, distinct on both GitHub themes.
PALETTE: Tuple[str, ...] = (
    "#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b",
    "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac",
)

THEMES: Dict[str, Dict[str, str]] = {
    "light": {"bg": "#ffffff", "fg": "#24292f", "muted": "#8c959f"},
    "dark": {"bg": "#0d1117", "fg": "#e6edf3", "muted": "#6e7681"},
}


@dataclass(frozen=True)
class CellType:
    """A synthetic cluster.

    Attributes:
        label: Text drawn above the cluster.
        center: Centroid in SVG pixel coordinates.
        spread: (sx, sy) standard deviation in pixels.
        n: Number of cells.
        color: Fill color.
    """

    label: str
    center: Tuple[float, float]
    spread: Tuple[float, float]
    n: int
    color: str


@dataclass(frozen=True)
class Embedding:
    """Cells already placed in canvas coordinates.

    Attributes:
        coords: (n, 2) array of final positions in SVG pixels.
        labels: Per-cell label, length n.
        colors: Label -> fill color.
    """

    coords: np.ndarray
    labels: List[str]
    colors: Dict[str, str]


# Nephron segments laid out roughly along the nephron axis.
KIDNEY_TYPES: Tuple[CellType, ...] = (
    CellType("POD", (110, 110), (16, 14), 50, "#e45756"),
    CellType("PT", (250, 200), (46, 28), 190, "#4c78a8"),
    CellType("TAL", (420, 130), (30, 20), 100, "#f58518"),
    CellType("DCT", (540, 220), (22, 18), 60, "#54a24b"),
    CellType("CD-PC", (660, 140), (26, 20), 70, "#b279a2"),
    CellType("EC", (330, 320), (34, 14), 70, "#72b7b2"),
    CellType("IMM", (640, 310), (20, 14), 40, "#9d755d"),
)


def _clip_to_box(pts: np.ndarray) -> np.ndarray:
    out = pts.copy()
    out[:, 0] = np.clip(out[:, 0], PAD_X, WIDTH - PAD_X)
    out[:, 1] = np.clip(out[:, 1], PAD_TOP, HEIGHT - PAD_BOTTOM)
    return out


def synthetic_embedding(
    types: Sequence[CellType], rng: np.random.Generator
) -> Embedding:
    """Sample a clustered embedding from hand-placed cell types.

    Raises:
        ValueError: On empty input, non-positive counts, or off-canvas centers.
    """
    if not types:
        raise ValueError("at least one CellType is required")
    pts: List[np.ndarray] = []
    labels: List[str] = []
    for ct in types:
        if ct.n <= 0:
            raise ValueError(f"{ct.label}: n must be positive, got {ct.n}")
        x, y = ct.center
        if not (0 <= x <= WIDTH and 0 <= y <= HEIGHT):
            raise ValueError(f"{ct.label}: center {ct.center} is off-canvas")
        pts.append(rng.normal(ct.center, ct.spread, size=(ct.n, 2)))
        labels.extend([ct.label] * ct.n)
    return Embedding(
        coords=_clip_to_box(np.vstack(pts)),
        labels=labels,
        colors={ct.label: ct.color for ct in types},
    )


def fit_to_canvas(xy: np.ndarray) -> np.ndarray:
    """Scale raw embedding coordinates into the plot box.

    Aspect ratio is preserved because stretching a UMAP distorts the relative
    cluster distances viewers instinctively read. Y is flipped since SVG y
    grows downward.
    """
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
        raise ValueError(f"expected a non-empty (n, 2) array, got shape {xy.shape}")
    if not np.isfinite(xy).all():
        raise ValueError("embedding contains NaN or inf")
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    box_w = WIDTH - 2 * PAD_X
    box_h = HEIGHT - PAD_TOP - PAD_BOTTOM
    scale = min(box_w / span[0], box_h / span[1])
    centered = (xy - (lo + hi) / 2) * scale
    out = np.empty_like(centered, dtype=float)
    out[:, 0] = WIDTH / 2 + centered[:, 0]
    out[:, 1] = PAD_TOP + box_h / 2 - centered[:, 1]
    return out


def load_embedding_csv(path: Path) -> Embedding:
    """Read an ``x,y,label`` CSV into canvas coordinates.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: On missing columns, non-numeric coordinates, an empty
            file, or more labels than the palette can distinguish.
    """
    if not path.is_file():
        raise FileNotFoundError(f"embedding CSV not found: {path}")
    xs: List[Tuple[float, float]] = []
    labels: List[str] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"x", "y", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        for i, row in enumerate(reader, start=2):
            try:
                xs.append((float(row["x"]), float(row["y"])))
            except ValueError as exc:
                raise ValueError(f"{path}:{i}: non-numeric coordinate") from exc
            labels.append(row["label"].strip())
    if not xs:
        raise ValueError(f"{path}: no rows")
    uniq = sorted(set(labels))
    if len(uniq) > MAX_LABELS:
        raise ValueError(
            f"{path}: {len(uniq)} labels; at most {MAX_LABELS} stay distinguishable"
        )
    # Sorted order keeps colors stable across regenerations.
    colors = {lab: PALETTE[i] for i, lab in enumerate(uniq)}
    return Embedding(fit_to_canvas(np.asarray(xs)), labels, colors)


def _circle(
    x0: float, y0: float, x1: float, y1: float, color: str, delay: float
) -> str:
    # Hold noise, ease into cluster, hold, ease back. Per-cell jitter in
    # keyTimes staggers the motion so clusters condense rather than snap.
    times = f"0;{0.08 + delay:.3f};{0.40 + delay:.3f};0.88;1"
    spline = "0 0 1 1;0.4 0 0.2 1;0 0 1 1;0.4 0 0.2 1"
    anim = (
        f'calcMode="spline" keyTimes="{times}" keySplines="{spline}" '
        f'dur="{DURATION_S}s" repeatCount="indefinite"'
    )
    return (
        f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="2.6" fill="{color}" '
        f'fill-opacity="0.85">'
        f'<animate attributeName="cx" '
        f'values="{x0:.1f};{x0:.1f};{x1:.1f};{x1:.1f};{x0:.1f}" {anim}/>'
        f'<animate attributeName="cy" '
        f'values="{y0:.1f};{y0:.1f};{y1:.1f};{y1:.1f};{y0:.1f}" {anim}/>'
        f"</circle>"
    )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _label(text: str, x: float, y: float, fg: str, bg: str) -> str:
    # Labels show only while clusters are formed; the reveal is the point.
    # A background-colored halo keeps them legible over dense dots.
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="13" font-weight="600" fill="{fg}" opacity="0" '
        f'paint-order="stroke" stroke="{bg}" stroke-width="3" stroke-opacity="0.85">'
        f"{_escape(text)}"
        f'<animate attributeName="opacity" values="0;0;1;1;0;0" '
        f'keyTimes="0;0.46;0.54;0.82;0.88;1" dur="{DURATION_S}s" '
        f'repeatCount="indefinite"/></text>'
    )


def _label_anchor(pts: np.ndarray) -> Tuple[float, float]:
    # Median plus a robust spread puts the label just above the dense core,
    # ignoring stray cells that would otherwise push it far away.
    cx, cy = np.median(pts, axis=0)
    top = np.percentile(pts[:, 1], 15)
    return float(cx), float(max(top - 8, PAD_TOP - 14))


def build_svg(
    embedding: Optional[Embedding] = None,
    theme: str = "light",
    seed: int = 7,
    caption: Optional[str] = None,
) -> str:
    """Build the animated SVG document.

    Args:
        embedding: Cells in canvas coordinates; synthetic kidney if None.
        theme: "light" or "dark" to match GitHub's color scheme.
        seed: RNG seed so the asset is byte-identical across regenerations.
        caption: Bottom-right text; defaults to a cell count.

    Returns:
        SVG markup.

    Raises:
        ValueError: If the theme or embedding is invalid.
    """
    if theme not in THEMES:
        raise ValueError(f"theme must be one of {sorted(THEMES)}, got {theme!r}")
    pal = THEMES[theme]
    rng = np.random.default_rng(seed)
    emb = embedding or synthetic_embedding(KIDNEY_TYPES, rng)
    n = len(emb.labels)
    if emb.coords.shape != (n, 2):
        raise ValueError("coords and labels length mismatch")

    start = _clip_to_box(
        rng.normal(
            (WIDTH / 2, (PAD_TOP + HEIGHT - PAD_BOTTOM) / 2),
            (WIDTH / 5, HEIGHT / 6),
            size=(n, 2),
        )
    )
    delays = rng.uniform(0.0, 0.05, size=n)
    circles = "".join(
        _circle(s[0], s[1], e[0], e[1], emb.colors[lab], float(d))
        for s, e, lab, d in zip(start, emb.coords, emb.labels, delays)
    )

    labels_arr = np.asarray(emb.labels)
    label_svg = "".join(
        _label(lab, *_label_anchor(emb.coords[labels_arr == lab]), pal["fg"], pal["bg"])
        for lab in emb.colors
        if (labels_arr == lab).any()
    )

    text = _escape(caption or f"{n} cells · noise → cell types")
    cap = (
        f'<text x="{WIDTH - 14}" y="{HEIGHT - 14}" text-anchor="end" '
        f'font-family="{FONT}" font-size="11" fill="{pal["muted"]}">{text}</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" '
        f'aria-label="Animated embedding of cells resolving into labeled clusters">'
        f'<rect width="100%" height="100%" rx="12" fill="{pal["bg"]}"/>'
        f"{circles}{label_svg}{cap}</svg>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--theme", choices=sorted(THEMES), default="light")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embedding", type=Path, help="CSV with x,y,label")
    parser.add_argument("--caption")
    args = parser.parse_args()
    try:
        emb = load_embedding_csv(args.embedding) if args.embedding else None
        svg = build_svg(emb, theme=args.theme, seed=args.seed, caption=args.caption)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(svg, encoding="utf-8")
    except (ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"wrote {args.out} ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
