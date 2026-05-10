import os
import tempfile

import pandas as pd
import pytest

from multiWellAnalysis.analysis.labels import loadLabels


def _writeLayout(plateDir, name, df):
    p = os.path.join(plateDir, name)
    df.to_csv(p, index=False)
    return p


def test_loadLabels_layout_default_col_index_1():
    with tempfile.TemporaryDirectory() as plateDir:
        _writeLayout(plateDir, 'plate1_layout.csv', pd.DataFrame({
            'wellId': ['A1', 'A2', 'B3'],
            'mutant': ['WT', 'dgrA', 'WT'],
            'media': ['LB', 'LB', 'M9'],
        }))
        labels = loadLabels(plateDir=plateDir)
    assert labels == {'A1': 'WT', 'A2': 'dgrA', 'B3': 'WT'}


def test_loadLabels_layout_pick_column():
    with tempfile.TemporaryDirectory() as plateDir:
        _writeLayout(plateDir, 'layoutThing.csv', pd.DataFrame({
            'wellId': ['A1', 'A2'],
            'mutant': ['WT', 'dgrA'],
            'media': ['LB', 'M9'],
        }))
        labels = loadLabels(plateDir=plateDir, columnName='media')
    assert labels == {'A1': 'LB', 'A2': 'M9'}


def test_loadLabels_layout_filename_match_is_case_insensitive():
    with tempfile.TemporaryDirectory() as plateDir:
        _writeLayout(plateDir, 'PlateLAYOUT.CSV', pd.DataFrame({
            'wellId': ['A1'], 'mutant': ['WT'],
        }))
        labels = loadLabels(plateDir=plateDir)
    assert labels == {'A1': 'WT'}


def test_loadLabels_layout_dropsNaLabels():
    with tempfile.TemporaryDirectory() as plateDir:
        _writeLayout(plateDir, 'layout.csv', pd.DataFrame({
            'wellId': ['A1', 'A2'],
            'mutant': ['WT', None],
        }))
        labels = loadLabels(plateDir=plateDir)
    assert labels == {'A1': 'WT'}


def test_loadLabels_layout_unknownColumnRaises():
    with tempfile.TemporaryDirectory() as plateDir:
        _writeLayout(plateDir, 'layout.csv', pd.DataFrame({
            'wellId': ['A1'], 'mutant': ['WT'],
        }))
        with pytest.raises(ValueError, match='Column'):
            loadLabels(plateDir=plateDir, columnName='nonexistent')


def test_loadLabels_layout_singleColumnRaises():
    with tempfile.TemporaryDirectory() as plateDir:
        _writeLayout(plateDir, 'layout.csv', pd.DataFrame({'wellId': ['A1']}))
        with pytest.raises(ValueError, match='only 1 column'):
            loadLabels(plateDir=plateDir)


def test_loadLabels_falls_back_to_conditions_when_no_layout():
    with tempfile.TemporaryDirectory() as plateDir:
        labels = loadLabels(plateDir=plateDir,
                            conditionsMap={'WT': ['A1', 'A2'], 'mut': ['B3']})
    assert labels == {'A1': 'WT', 'A2': 'WT', 'B3': 'mut'}


def test_loadLabels_layout_takes_precedence_over_conditions():
    with tempfile.TemporaryDirectory() as plateDir:
        _writeLayout(plateDir, 'layout.csv', pd.DataFrame({
            'wellId': ['A1'], 'mutant': ['fromLayout'],
        }))
        labels = loadLabels(plateDir=plateDir,
                            conditionsMap={'fromConditions': ['A1']})
    assert labels == {'A1': 'fromLayout'}


def test_loadLabels_returns_empty_when_no_source():
    assert loadLabels() == {}


def test_loadLabels_picks_most_recent_layout_when_multiple():
    with tempfile.TemporaryDirectory() as plateDir:
        old = _writeLayout(plateDir, 'old_layout.csv',
                           pd.DataFrame({'wellId': ['A1'], 'mutant': ['old']}))
        new = _writeLayout(plateDir, 'new_layout.csv',
                           pd.DataFrame({'wellId': ['A1'], 'mutant': ['new']}))
        # ensure mtime ordering
        os.utime(old, (1, 1))
        os.utime(new, (2, 2))
        labels = loadLabels(plateDir=plateDir)
    assert labels == {'A1': 'new'}
