import numpy as np
import pandas as pd
import pytest

from multiWellAnalysis.analysis.embedding import (
    applyBiomassFloor,
    fitUmapGrid,
    selectUmapFeatures,
)


def _wideRow(plate, well, biomassValues, **extra):
    row = {'plateId': plate, 'wellId': well}
    for i, v in enumerate(biomassValues):
        row[f'biomass_t{i}'] = v
    row.update(extra)
    return row


def _syntheticWide(nWells=40, nFrames=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(nWells):
        plate = f'p{i // 10}'
        well = f'A{i % 96 + 1}'
        # two clusters separated in feature space
        offset = 0.0 if i < nWells // 2 else 1.0
        biomass = (rng.random(nFrames) * 0.1 + 0.05 + offset * 0.5).tolist()
        haralick = (rng.random(nFrames) + offset * 5).tolist()
        skew = rng.random(nFrames).tolist()  # should be filtered out
        row = _wideRow(plate, well, biomass)
        for j, v in enumerate(haralick):
            row[f'whole_haralick_entropy_t{j}'] = v
        for j, v in enumerate(skew):
            row[f'colony_meanIntensity_skew_t{j}'] = v
        rows.append(row)
    return pd.DataFrame(rows)


def test_selectUmapFeatures_keeps_biomass_and_whole_entropy_haralick():
    df = pd.DataFrame(columns=[
        'plateId', 'wellId',
        'biomass_t0', 'biomass_t1',
        'whole_haralick_entropy_t0',
        'whole_haralick_contrast_t0',
        'whole_intensity_mean_t0',
        'whole_intensity_skew_t0',
        'colony_area_mean_t0',
        'colAgg_meanIntensity_skew_t0',
        'whole_haralick_entropy_skew_t0',  # ends in _skew → filter out
    ])
    sel = selectUmapFeatures(df)
    assert 'biomass_t0' in sel
    assert 'biomass_t1' in sel
    assert 'whole_haralick_entropy_t0' in sel
    assert 'whole_haralick_contrast_t0' in sel  # has 'haralick'
    assert 'whole_intensity_mean_t0' not in sel  # no entropy/haralick
    assert 'whole_haralick_entropy_skew_t0' not in sel  # _skew suffix wins
    assert 'colony_area_mean_t0' not in sel  # colony_ excluded
    assert 'plateId' not in sel and 'wellId' not in sel


def test_applyBiomassFloor_splits_rows():
    df = pd.DataFrame([
        {'plateId': 'p1', 'wellId': 'A1', 'biomass_t0': 0.001, 'biomass_t1': 0.002},
        {'plateId': 'p1', 'wellId': 'A2', 'biomass_t0': 0.003, 'biomass_t1': 0.010},
    ])
    kept, excluded = applyBiomassFloor(df, floor=0.005)
    assert kept['wellId'].tolist() == ['A2']
    assert excluded['wellId'].tolist() == ['A1']


def test_fitUmapGrid_returns_full_3x3_grid():
    df = _syntheticWide(nWells=40)
    embeddings, excluded, scaler, reducers, featureCols = fitUmapGrid(df)
    # 9 (nn, md) pairs
    assert len(reducers) == 9
    assert set(reducers.keys()) == {(nn, md) for nn in (10, 20, 30) for md in (0.1, 0.2, 0.3)}
    # each pair contributes 2 cols (x, y), plus 2 id cols
    assert embeddings.shape == (40, 2 + 2 * 9)
    for nn in (10, 20, 30):
        for md in (0.1, 0.2, 0.3):
            assert f'umap_x_nn{nn}_md{md}' in embeddings.columns
            assert f'umap_y_nn{nn}_md{md}' in embeddings.columns
    assert excluded.empty
    assert all(c.startswith(('biomass_', 'whole_haralick_')) for c in featureCols)


def test_fitUmapGrid_excludes_low_biomass_rows():
    df = _syntheticWide(nWells=40)
    # zero-out biomass for first 5 rows so they fail the floor
    biomassCols = [c for c in df.columns if c.startswith('biomass_t')]
    df.loc[:4, biomassCols] = 0.0
    embeddings, excluded, _, _, _ = fitUmapGrid(df)
    assert len(embeddings) == 35
    assert len(excluded) == 5


def test_fitUmapGrid_clamps_nNeighbors_for_small_inputs():
    df = _syntheticWide(nWells=8)
    with pytest.warns(UserWarning, match='Clamping'):
        embeddings, _, _, reducers, _ = fitUmapGrid(df)
    assert len(embeddings) == 8
    # all nn values get clamped to 7
    for (nn, md), reducer in reducers.items():
        assert reducer.n_neighbors == 7


def test_fitUmapGrid_raises_when_no_features():
    df = pd.DataFrame([
        {'plateId': 'p1', 'wellId': 'A1', 'colony_area_mean_t0': 1.0, 'biomass_t0': 0.01},
    ] * 30)
    # only colony_ which is filtered out — but biomass_t0 keeps it from being empty
    # so let's drop biomass too
    df = df.drop(columns=['biomass_t0'])
    with pytest.raises(ValueError, match='No columns'):
        fitUmapGrid(df)


def test_fitUmapGrid_raises_when_all_excluded():
    df = _syntheticWide(nWells=10)
    biomassCols = [c for c in df.columns if c.startswith('biomass_t')]
    df[biomassCols] = 0.0
    with pytest.raises(ValueError, match='biomass floor'):
        fitUmapGrid(df)
