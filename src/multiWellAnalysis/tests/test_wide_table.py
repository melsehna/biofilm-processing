import os
import tempfile

import pandas as pd
import pytest

from multiWellAnalysis.analysis.wide_table import buildWideTable


def _writeMaster(tmpDir, rows):
    path = os.path.join(tmpDir, 'master_frame_features.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(plate, well, mag, frame, **feats):
    base = {'drawerID': 'd1', 'plateID': plate, 'wellID': well, 'mag': mag, 'frame': frame}
    base.update(feats)
    return base


def test_buildWideTable_pivots_per_well_per_frame():
    with tempfile.TemporaryDirectory() as td:
        rows = []
        for plate in ['p1', 'p2']:
            for well in ['A1', 'A2']:
                for f in range(3):
                    rows.append(_row(plate, well, '_03', f, biomass=0.1 * f))
        wide = buildWideTable(_writeMaster(td, rows), magnification='_03')
    assert wide.shape == (4, 5)  # plateId, wellId, biomass_t0/1/2
    assert set(wide.columns) == {'plateId', 'wellId', 'biomass_t0', 'biomass_t1', 'biomass_t2'}


def test_buildWideTable_filters_by_mag():
    with tempfile.TemporaryDirectory() as td:
        rows = [
            _row('p1', 'A1', '_03', 0, biomass=1.0),
            _row('p1', 'A1', '_02', 0, biomass=99.0),
        ]
        wide = buildWideTable(_writeMaster(td, rows), magnification='_03')
    assert len(wide) == 1
    assert wide['biomass_t0'].iloc[0] == 1.0


def test_buildWideTable_renames_colAgg_and_skewness():
    with tempfile.TemporaryDirectory() as td:
        rows = [_row('p1', 'A1', '_03', 0,
                     colAgg_meanIntensity_skewness=0.5,
                     colAgg_nColonies=3)]
        wide = buildWideTable(_writeMaster(td, rows), magnification='_03')
    assert 'colony_meanIntensity_skew_t0' in wide.columns
    assert 'nColonies_t0' in wide.columns


def test_buildWideTable_frames_range_filter():
    with tempfile.TemporaryDirectory() as td:
        rows = [_row('p1', 'A1', '_03', f, biomass=float(f)) for f in range(5)]
        wide = buildWideTable(_writeMaster(td, rows), magnification='_03',
                              framesRange=(1, 3))
    assert set(wide.columns) == {'plateId', 'wellId', 'biomass_t1', 'biomass_t2', 'biomass_t3'}


def test_buildWideTable_raises_when_mag_absent():
    with tempfile.TemporaryDirectory() as td:
        rows = [_row('p1', 'A1', '_03', 0, biomass=1.0)]
        with pytest.raises(ValueError, match='No rows'):
            buildWideTable(_writeMaster(td, rows), magnification='_99')
