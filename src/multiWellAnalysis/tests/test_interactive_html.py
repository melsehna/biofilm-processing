"""End-to-end test: real UMAP fit -> HTML viewer -> sanity-check the JSON."""
import json
import os
import re
import tempfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('umap', reason='install with: pip install -e .[umap]')

from multiWellAnalysis.analysis.embedding import fitUmapGrid
from multiWellAnalysis.analysis.interactive_html import writeInteractiveHtml


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


def _extractJsonBlock(html, varName):
    """Pull the literal value assigned to `const <varName> = ...;` from the JS."""
    m = re.search(rf'const {varName} = (.*?);', html, re.DOTALL)
    assert m, f'{varName} not found in HTML'
    return json.loads(m.group(1))


def test_writeInteractiveHtml_writes_html_with_data_block():
    emb = _fitOnce()
    labels = {f'A{i + 1}': ('WT' if i < 10 else 'mut') for i in range(40)}
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'sub', 'umap.html')
        writeInteractiveHtml(emb, labels, out, title='test')
        html = open(out).read()
    assert '<canvas id="canvas"></canvas>' in html
    assert '<title>test</title>' in html

    data = _extractJsonBlock(html, 'allData')
    # 9 (nn, md) pairs from the default grid
    assert len(data) == 9
    assert '20_0.2' in data
    pts = data['20_0.2']
    assert len(pts) == 40
    sample = pts[0]
    for key in ('x', 'y', 'plate', 'well', 'label', 'mp4'):
        assert key in sample
    # labels propagated
    wt_count = sum(1 for p in pts if p['label'] == 'WT')
    assert wt_count == 10


def test_writeInteractiveHtml_with_no_labels():
    emb = _fitOnce()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'no_labels.html')
        writeInteractiveHtml(emb, {}, out)
        html = open(out).read()
    data = _extractJsonBlock(html, 'allData')
    pts = data['20_0.2']
    assert all(p['label'] == '' for p in pts)


def test_writeInteractiveHtml_includes_mp4_paths_when_present():
    emb = _fitOnce()
    emb = emb.copy()
    emb['mp4'] = emb['wellId'].apply(lambda w: f'../plate1/processedImages/{w}_overlay.mp4')
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'with_mp4.html')
        writeInteractiveHtml(emb, {}, out)
        html = open(out).read()
    data = _extractJsonBlock(html, 'allData')
    sample = data['20_0.2'][0]
    assert sample['mp4'].endswith('_overlay.mp4')
    assert sample['mp4'].startswith('../plate1/')


def test_writeInteractiveHtml_paramCombos_match_embedding_columns():
    emb = _fitOnce()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'params.html')
        writeInteractiveHtml(emb, {}, out)
        html = open(out).read()
    params = _extractJsonBlock(html, 'paramCombos')
    assert sorted(params) == [[10, 0.1], [10, 0.2], [10, 0.3],
                              [20, 0.1], [20, 0.2], [20, 0.3],
                              [30, 0.1], [30, 0.2], [30, 0.3]]
