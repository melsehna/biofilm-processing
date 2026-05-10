"""End-to-end test of the static plot path: real fit -> real PNGs on disk."""
import os
import tempfile

import numpy as np
import pandas as pd

from multiWellAnalysis.analysis.embedding import fitUmapGrid
from multiWellAnalysis.analysis.static_plot import plotGrid, plotStatic


def _syntheticWide(nWells=40, nFrames=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(nWells):
        offset = 0.0 if i < nWells // 2 else 1.0
        biomass = (rng.random(nFrames) * 0.1 + 0.05 + offset * 0.5).tolist()
        haralick = (rng.random(nFrames) + offset * 5).tolist()
        row = {'plateId': f'p{i // 10}', 'wellId': f'A{i + 1}'}
        for j, v in enumerate(biomass):
            row[f'biomass_t{j}'] = v
        for j, v in enumerate(haralick):
            row[f'whole_haralick_entropy_t{j}'] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _fitOnce():
    df = _syntheticWide(nWells=40)
    embeddings, _, _, _, _ = fitUmapGrid(df)
    return embeddings


def test_plotStatic_writes_png_with_labels():
    emb = _fitOnce()
    labels = {f'A{i + 1}': ('WT' if i < 10 else f'mut{i % 4}') for i in range(40)}
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'sub', 'static.png')
        result = plotStatic(emb, labels, out)
    assert result == out


def test_plotStatic_runs_with_no_labels():
    emb = _fitOnce()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'no_labels.png')
        plotStatic(emb, {}, out)
        assert os.path.getsize(out) > 0


def test_plotGrid_writes_3x3_png():
    emb = _fitOnce()
    labels = {f'A{i + 1}': ('WT' if i < 10 else 'mut') for i in range(40)}
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'grid.png')
        plotGrid(emb, labels, out, title='test grid')
        assert os.path.getsize(out) > 1000  # non-trivial PNG


def test_plotStatic_handles_missing_embedding_pair():
    emb = _fitOnce()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'missing.png')
        plotStatic(emb, {}, out, nn=99, md=0.99)  # not in grid
        # function still writes a PNG with a "no fit" placeholder
        assert os.path.getsize(out) > 0
