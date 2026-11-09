import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from make_umap_svg import (
    HEIGHT,
    KIDNEY_TYPES,
    PAD_BOTTOM,
    CellType,
    build_svg,
    fit_to_canvas,
    load_embedding_csv,
    synthetic_embedding,
)

NS = "{http://www.w3.org/2000/svg}"


def _write_csv(path: Path, n_per: int = 30) -> Path:
    rng = np.random.default_rng(0)
    rows = ["x,y,label"]
    for lab, c in [("B cells", (0, 0)), ("NK cells", (5, 5)), ("CD4 T", (-4, 6))]:
        for x, y in rng.normal(c, 0.5, size=(n_per, 2)):
            rows.append(f"{x},{y},{lab}")
    path.write_text("\n".join(rows) + "\n")
    return path


def test_synthetic_svg_has_one_circle_per_cell() -> None:
    root = ET.fromstring(build_svg(theme="dark"))
    assert len(root.findall(f"{NS}circle")) == sum(ct.n for ct in KIDNEY_TYPES)


def test_static_positions_match_cluster_state() -> None:
    # Non-SMIL renderers must show clusters, not a pile at (0, 0).
    root = ET.fromstring(build_svg())
    assert all(float(c.get("cx")) > 0 for c in root.findall(f"{NS}circle"))


def test_cells_stay_clear_of_caption_band() -> None:
    root = ET.fromstring(build_svg())
    ys = [float(c.get("cy")) for c in root.findall(f"{NS}circle")]
    assert max(ys) <= HEIGHT - PAD_BOTTOM


def test_seed_makes_output_reproducible() -> None:
    assert build_svg(seed=1) == build_svg(seed=1)


def test_csv_embedding_round_trip(tmp_path: Path) -> None:
    emb = load_embedding_csv(_write_csv(tmp_path / "e.csv"))
    root = ET.fromstring(build_svg(emb, caption="pbmc3k & friends"))
    assert len(root.findall(f"{NS}circle")) == 90
    texts = {t.text for t in root.findall(f"{NS}text")}
    assert {"B cells", "NK cells", "CD4 T", "pbmc3k & friends"} <= texts


def test_fit_to_canvas_flips_y() -> None:
    out = fit_to_canvas(np.array([[0.0, 0.0], [0.0, 1.0]]))
    assert out[1, 1] < out[0, 1]  # larger UMAP y is higher on screen


def test_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="theme"):
        build_svg(theme="neon")
    with pytest.raises(ValueError, match="off-canvas"):
        synthetic_embedding(
            [CellType("X", (9999, 0), (1, 1), 5, "#000")], np.random.default_rng()
        )
    bad = tmp_path / "bad.csv"
    bad.write_text("x,label\n1,a\n")
    with pytest.raises(ValueError, match="missing column"):
        load_embedding_csv(bad)
    with pytest.raises(ValueError, match="NaN"):
        fit_to_canvas(np.array([[np.nan, 0.0]]))
