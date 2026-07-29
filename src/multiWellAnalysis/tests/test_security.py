"""Regression tests for untrusted-input handling.

Sources: `*layout*.csv` sidecars, plate/well directory names, and Cytation TIFF
headers all come from shared storage and are not authored by the pipeline.
"""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest
import tifffile

from multiWellAnalysis.analysis.interactive_html import writeInteractiveHtml
from multiWellAnalysis.gui.tabs.run import ProcessingWorker
from multiWellAnalysis.processing.image_metadata import readCytationMeta

BREAKOUT = '</script><img src=x onerror=alert(1)>'


def _embeddings():
    return pd.DataFrame({
        'plateId': ['p1'], 'wellId': ['A1'],
        'umap_x_nn20_md0.2': [0.1], 'umap_y_nn20_md0.2': [0.3],
    })


def _writeViewer(tmp_path, labels, **kw):
    out = tmp_path / 'v.html'
    writeInteractiveHtml(_embeddings(), labels, str(out), **kw)
    return out.read_text()


def test_label_cannot_close_the_script_block(tmp_path):
    html = _writeViewer(tmp_path, {'A1': BREAKOUT})
    assert BREAKOUT not in html
    body = html[html.index('<script>') + len('<script>'):]
    # the first closer must be the template's own, not one smuggled in via data
    assert not body[body.index('</script>'):].startswith(BREAKOUT)


def test_escaped_payload_still_round_trips(tmp_path):
    """Escaping must not corrupt the value — labels still render correctly."""
    html = _writeViewer(tmp_path, {'A1': BREAKOUT})
    data = json.loads(re.search(r'^const allData = (.*);$', html, re.M).group(1))
    assert data['20_0.2'][0]['label'] == BREAKOUT


def test_title_is_html_escaped(tmp_path):
    html = _writeViewer(tmp_path, {}, title='<script>x</script>')
    assert '<title>UMAP' not in html
    assert '&lt;script&gt;' in html


def test_innerhtml_sinks_are_escaped(tmp_path):
    html = _writeViewer(tmp_path, {'A1': 'wt'})
    for raw in ('${p.label}', '${p.plate}', '${p.well}', '${p.mp4}'):
        assert raw not in html
    assert 'const esc =' in html


def _tif(tmp_path, description):
    p = tmp_path / 'a.tif'
    tifffile.imwrite(str(p), np.zeros((4, 4), np.uint16), description=description)
    return str(p)


def test_entity_expansion_is_refused(tmp_path):
    entities = ''.join(
        '<!ENTITY e%d "%s">' % (i, ('&e%d;' % (i - 1)) * 10 if i else 'x')
        for i in range(6))
    xml = f'<!DOCTYPE r [{entities}]><BTIImageMetaData><ImageAcquisition>' \
          f'<ObjectiveSize>&e5;</ObjectiveSize></ImageAcquisition></BTIImageMetaData>'
    with pytest.raises(ValueError, match='DTD or entity'):
        readCytationMeta(_tif(tmp_path, xml))


def test_benign_metadata_still_parses(tmp_path):
    xml = ('<BTIImageMetaData><ImageAcquisition>'
           '<ObjectiveSize>10</ObjectiveSize><PixelWidth>1992</PixelWidth>'
           '<ImageWidthMicrons>1389.0</ImageWidthMicrons>'
           '</ImageAcquisition></BTIImageMetaData>')
    meta = readCytationMeta(_tif(tmp_path, xml))
    assert meta['objective'] == 10
    assert meta['pxToUm'] == pytest.approx(1389.0 / 1992)


@pytest.mark.parametrize('rel, expected', [
    ('', True),              # the root itself (auto-staging teardown)
    ('plateA', True),
    ('..', False),
    ('_sibling', False),     # shared prefix, not a child
])
def test_delete_containment(tmp_path, rel, expected):
    root = tmp_path / 'staging'
    (root / 'plateA').mkdir(parents=True)
    (tmp_path / 'staging_sibling').mkdir()
    target = str(root) + rel if rel == '_sibling' else str(root / rel)
    assert ProcessingWorker._isContainedIn(target, str(root)) is expected


def test_delete_containment_rejects_symlink_escape(tmp_path):
    root = tmp_path / 'staging'
    root.mkdir()
    outside = tmp_path / 'elsewhere'
    outside.mkdir()
    os.symlink(str(outside), str(root / 'link'))
    assert ProcessingWorker._isContainedIn(str(root / 'link'), str(root)) is False


def test_delete_containment_rejects_unset_and_filesystem_root(tmp_path):
    assert ProcessingWorker._isContainedIn(str(tmp_path), '') is False
    assert ProcessingWorker._isContainedIn(str(tmp_path), '/') is False
