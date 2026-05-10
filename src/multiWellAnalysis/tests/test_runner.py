"""End-to-end test of runUmap: synthetic master CSV -> all artifacts on disk."""
import json
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('umap', reason='install with: pip install -e .[umap]')

from multiWellAnalysis.analysis.runner import runUmap


def _writeMaster(outputRoot, nPlates=2, wellsPerPlate=20, nFrames=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(nPlates):
        for w in range(wellsPerPlate):
            offset = 0.0 if w < wellsPerPlate // 2 else 1.0
            for f in range(nFrames):
                rows.append({
                    'drawerID': 'd1',
                    'plateID': f'plate{p}',
                    'wellID': f'A{w + 1}',
                    'mag': '_03',
                    'frame': f,
                    'biomass': 0.05 + offset * 0.5 + rng.random() * 0.05,
                    'whole_haralick_entropy': 1.0 + offset * 5 + rng.random(),
                })
    path = os.path.join(outputRoot, 'master_frame_features.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _makePlateDir(parent, plateId, layoutDf=None):
    plateDir = os.path.join(parent, f'{plateId}_raw')
    os.makedirs(plateDir, exist_ok=True)
    if layoutDf is not None:
        layoutDf.to_csv(os.path.join(plateDir, f'{plateId}_layout.csv'), index=False)
    return plateDir


def _makeOverlay(outputRoot, plateId, wellId):
    procDir = os.path.join(outputRoot, plateId, 'processedImages')
    os.makedirs(procDir, exist_ok=True)
    p = os.path.join(procDir, f'{wellId}_overlay.mp4')
    with open(p, 'wb') as f:
        f.write(b'fake mp4')
    return p


def test_runUmap_writes_all_artifacts():
    with tempfile.TemporaryDirectory() as td:
        outputRoot = os.path.join(td, 'run_out')
        os.makedirs(outputRoot)
        _writeMaster(outputRoot, nPlates=2, wellsPerPlate=20)

        out = runUmap(outputRoot, magnification='_03',
                      doStatic=True, doInteractive=True)

    assert 'embeddings' in out and out['embeddings'].endswith('umap_10X_embeddings.parquet')
    assert 'scaler' in out and out['scaler'].endswith('umap_10X_scaler.pkl')
    assert 'reducers' in out and out['reducers'].endswith('umap_10X_reducers.pkl')
    assert 'features' in out and out['features'].endswith('umap_10X_features.json')
    assert 'static' in out and out['static'].endswith('umap_10X_static.png')
    assert 'grid' in out and out['grid'].endswith('umap_10X_grid.png')
    assert 'interactive' in out and out['interactive'].endswith('umap_10X_interactive.html')


def test_runUmap_uses_objective_label_when_plateMeta_provided():
    with tempfile.TemporaryDirectory() as td:
        outputRoot = os.path.join(td, 'run_out')
        os.makedirs(outputRoot)
        _writeMaster(outputRoot, nPlates=1, wellsPerPlate=30)
        plateMeta = {'/raw/plate0': {'_03': {'objective': 10, 'pxToUm': 0.7}}}

        out = runUmap(outputRoot, magnification='_03',
                      doStatic=True, doInteractive=False, plateMeta=plateMeta)

    assert 'umap_10X' in out['static']


def test_runUmap_resolves_labels_from_per_plate_layout_csv():
    with tempfile.TemporaryDirectory() as td:
        outputRoot = os.path.join(td, 'run_out')
        os.makedirs(outputRoot)
        _writeMaster(outputRoot, nPlates=2, wellsPerPlate=20)
        # plate0 has WT at A1, mut at A2; plate1 has the OPPOSITE — same well, different label
        plate0Dir = _makePlateDir(td, 'plate0', pd.DataFrame({
            'wellId': [f'A{i + 1}' for i in range(20)],
            'mutant': ['WT' if i < 10 else f'mut0_{i}' for i in range(20)],
        }))
        plate1Dir = _makePlateDir(td, 'plate1', pd.DataFrame({
            'wellId': [f'A{i + 1}' for i in range(20)],
            'mutant': ['fromPlate1' for _ in range(20)],
        }))
        plateDirMap = {'plate0': plate0Dir, 'plate1': plate1Dir}

        out = runUmap(outputRoot, magnification='_03',
                      doStatic=False, doInteractive=True,
                      plateDirMap=plateDirMap)

        html = open(out['interactive']).read()
        # plate1 wells should all be labeled 'fromPlate1' even though plate0 has different labels
        # at the same wellIds — verifies (plateId, wellId) tuple keying works
        assert '"plate": "plate1", "well": "A1", "label": "fromPlate1"' in html
        assert '"plate": "plate0", "well": "A1", "label": "WT"' in html


def test_runUmap_resolves_overlay_paths_when_files_exist():
    with tempfile.TemporaryDirectory() as td:
        outputRoot = os.path.join(td, 'run_out')
        os.makedirs(outputRoot)
        _writeMaster(outputRoot, nPlates=1, wellsPerPlate=30)
        # only A1 has an overlay
        _makeOverlay(outputRoot, 'plate0', 'A1')

        out = runUmap(outputRoot, magnification='_03',
                      doStatic=False, doInteractive=True)

        html = open(out['interactive']).read()
        assert '"mp4": "../plate0/processedImages/A1_overlay.mp4"' in html
        # other wells should have empty mp4 string
        assert '"well": "A2"' in html and '"mp4": ""' in html


def test_runUmap_persists_loadable_scaler_and_reducers():
    with tempfile.TemporaryDirectory() as td:
        outputRoot = os.path.join(td, 'run_out')
        os.makedirs(outputRoot)
        _writeMaster(outputRoot, nPlates=2, wellsPerPlate=20)

        out = runUmap(outputRoot, magnification='_03',
                      doStatic=False, doInteractive=False)

        with open(out['scaler'], 'rb') as f:
            scaler = pickle.load(f)
        with open(out['reducers'], 'rb') as f:
            reducers = pickle.load(f)
        with open(out['features']) as f:
            features = json.load(f)

        # scaler is fit and reducers is the (nn, md) -> UMAP dict
        assert hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_')
        assert len(reducers) == 9
        assert (20, 0.2) in reducers
        assert isinstance(features, list) and len(features) > 0


def test_runUmap_raises_when_master_csv_missing():
    with tempfile.TemporaryDirectory() as td:
        try:
            runUmap(td, magnification='_03')
        except FileNotFoundError as e:
            assert 'master_frame_features.csv' in str(e)
        else:
            raise AssertionError('expected FileNotFoundError')
