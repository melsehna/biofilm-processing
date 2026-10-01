"""Golden/regression tests for the core image + tracking pipeline.

These pin the numeric behavior the published results depend on — the code that
previously had no coverage (only the `analysis/` UMAP subpackage did). Inputs are
tiny hand-built arrays with known answers, so a change in a dependency
(skimage.measure.label, morphology, scipy.distance_transform_edt) or in the code
that shifts these answers fails loudly rather than silently drifting features.
"""
import os

# Guard against any transitive Qt import needing a display.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Bit-depth scaling — all inputs land on a common [0, 1] photometric axis so
# cross-well intensity math is comparable (analysis_main._toBitDepthScaled).
# ---------------------------------------------------------------------------

def test_bit_depth_scaling_paths():
    from multiWellAnalysis.processing.analysis_main import _toBitDepthScaled

    # integer dtype -> divide by the dtype full-scale
    u8 = np.array([0, 255], dtype=np.uint8)
    assert np.allclose(_toBitDepthScaled(u8), [0.0, 1.0])
    u16 = np.array([0, 65535], dtype=np.uint16)
    assert np.allclose(_toBitDepthScaled(u16), [0.0, 1.0])

    # float already in [0, 1] -> unchanged
    f = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    assert np.allclose(_toBitDepthScaled(f), f)

    # float with max > 1.5 -> infer 8-bit vs 16-bit full-scale
    assert np.allclose(_toBitDepthScaled(np.array([255.0])), [1.0])
    assert np.allclose(_toBitDepthScaled(np.array([65535.0])), [1.0])

    # None passes through (Imin/Imax may be absent)
    assert _toBitDepthScaled(None) is None

    # result is always float32
    assert _toBitDepthScaled(u8).dtype == np.float32


# ---------------------------------------------------------------------------
# Thresholding + dust correction (processing.segmentation).
# ---------------------------------------------------------------------------

def test_compute_mask_is_strict_threshold():
    from multiWellAnalysis.processing.segmentation import computeMaskInplace

    stack = np.array([[[0.03, 0.05]]], dtype=np.float32)  # (1, 1, 2)
    masks = np.zeros_like(stack, dtype=bool)
    computeMaskInplace(stack, masks, fixedThresh=0.04)
    # strictly greater-than: 0.03 -> off, 0.05 -> on
    assert not masks[0, 0, 0]
    assert masks[0, 0, 1]


def test_dust_correction_zeroes_transient_but_keeps_growth():
    from multiWellAnalysis.processing.segmentation import dustCorrectInplace

    # (H=1, W=3, T=3). Columns: persistent / dust (on@0 then off) / late growth.
    masks = np.zeros((1, 3, 3), dtype=bool)
    masks[0, 0, :] = [True, True, True]     # persistent -> kept
    masks[0, 1, :] = [True, False, True]    # on at t0, off at t1 -> dust, killed
    masks[0, 2, :] = [False, True, True]    # appears later -> real biofilm, kept

    dustCorrectInplace(masks)

    assert masks[0, 0].tolist() == [True, True, True]
    assert masks[0, 1].tolist() == [False, False, False]  # dust removed everywhere
    assert masks[0, 2].tolist() == [False, True, True]     # untouched


# ---------------------------------------------------------------------------
# Colony segmentation (colony.segmentation.segmentColonies): fill holes,
# drop sub-min-area specks, connected-component label.
# ---------------------------------------------------------------------------

def test_segment_colonies_counts_and_area_filter():
    from multiWellAnalysis.colony.segmentation import segmentColonies

    mask = np.zeros((20, 20), dtype=bool)
    mask[2:7, 2:7] = True     # 25 px blob -> kept
    mask[4, 4] = False        # interior hole -> filled, stays one object
    mask[12:17, 12:17] = True  # 25 px blob -> kept
    mask[0, 19] = True        # 1 px speck -> removed by min-area filter

    labels, props = segmentColonies(mask, mask, minColonyArea_px=10)

    assert labels.max() == 2          # two colonies, speck dropped
    assert len(props) == 2
    # hole was filled -> the first blob is a solid 25 px region
    assert sorted(p.area for p in props) == [25, 25]


# ---------------------------------------------------------------------------
# Cytation metadata parsing (image_metadata.readCytationMeta):
# pxToUm = ImageWidthMicrons / PixelWidth.
# ---------------------------------------------------------------------------

def _write_cytation_tif(path, objective=10, pixel_width=1992, width_um=1389.0):
    import tifffile

    xml = (
        f'<BTIImageMetaData><ImageAcquisition>'
        f'<ObjectiveSize>{objective}</ObjectiveSize>'
        f'<PixelWidth>{pixel_width}</PixelWidth>'
        f'<ImageWidthMicrons>{width_um}</ImageWidthMicrons>'
        f'</ImageAcquisition></BTIImageMetaData>'
    )
    # metadata=None so tag 270 (ImageDescription) holds exactly our XML.
    tifffile.imwrite(str(path), np.zeros((4, 4), np.uint16),
                     description=xml, metadata=None)


def test_read_cytation_meta_pxtoum(tmp_path):
    from multiWellAnalysis.processing.image_metadata import readCytationMeta

    p = tmp_path / 'A1_02_1_1_Bright Field_001.tif'
    _write_cytation_tif(p, objective=10, pixel_width=1992, width_um=1389.0)

    meta = readCytationMeta(str(p))
    assert meta['objective'] == 10
    assert meta['pxToUm'] == pytest.approx(1389.0 / 1992, rel=1e-6)


def test_read_cytation_meta_raises_on_zero_pixel_width(tmp_path):
    from multiWellAnalysis.processing.image_metadata import readCytationMeta

    p = tmp_path / 'bad.tif'
    _write_cytation_tif(p, pixel_width=0)
    with pytest.raises(ValueError):
        readCytationMeta(str(p))


# ---------------------------------------------------------------------------
# Seed-frame detection (runTrackingGUI.findSeedFrame): first frame with biomass
# >= threshold for `minConsecutive` frames.
# ---------------------------------------------------------------------------

def test_find_seed_frame():
    from multiWellAnalysis.colony.runTrackingGUI import findSeedFrame

    # rises above 0.005 and stays there from index 2
    assert findSeedFrame([0.0, 0.001, 0.006, 0.007, 0.008], threshold=0.005) == 2
    # a single spike (index 2) is not 2 consecutive -> no seed
    assert findSeedFrame([0.0, 0.001, 0.006, 0.001], threshold=0.005) is None


# ---------------------------------------------------------------------------
# Tracking propagation (runTrackingGUI.propagateLabelsFastVectorized): a colony
# that shifts within the effective radius keeps its label; a distinct far colony
# gets a fresh id. This is the property the persistent-footprint design protects.
# ---------------------------------------------------------------------------

def test_propagation_preserves_id_across_shift_and_allocates_new():
    from multiWellAnalysis.colony.runTrackingGUI import propagateLabelsFastVectorized

    labelsPrev = np.zeros((20, 20), dtype=np.int32)
    labelsPrev[2:6, 2:6] = 5             # existing colony, id 5

    maskNext = np.zeros((20, 20), dtype=bool)
    maskNext[3:7, 3:7] = True            # same colony, shifted by (1, 1)
    maskNext[14:18, 14:18] = True        # a new, far colony

    labelsNext, nextLabelId = propagateLabelsFastVectorized(
        labelsPrev, maskNext, nextLabelId=6, effectiveRadius=3, min_area=4)

    assert set(np.unique(labelsNext)) == {0, 5, 6}
    assert labelsNext[6, 6] == 5         # shifted colony kept its id, not relabeled
    assert labelsNext[15, 15] == 6       # far colony got the fresh id
    assert nextLabelId == 7              # counter advanced once


# ---------------------------------------------------------------------------
# Provenance stamping (master_csv.assembleMasterCsvs): a run-level
# provenance.json sidecar records which pipeline build produced the master CSVs.
# (Also the first coverage of the master-CSV assembly itself.)
# ---------------------------------------------------------------------------

def test_master_csv_writes_provenance_sidecar(tmp_path):
    import json
    import pandas as pd
    from multiWellAnalysis.processing.master_csv import assembleMasterCsvs

    # Minimal per-plate layout: processedImages/index.csv + one biomass CSV.
    proc = tmp_path / 'P1' / 'processedImages'
    proc.mkdir(parents=True)
    pd.DataFrame({'frame': [0, 1, 2], 'biomass': [0.0, 0.01, 0.02]}).to_csv(
        proc / 'A1_biomass.csv', index=False)
    pd.DataFrame([{'plate': 'P1', 'plate_path': str(tmp_path / 'P1'),
                   'well': 'A1', 'mag': '_03', 'biomass': 'A1_biomass.csv'}]).to_csv(
        proc / 'index.csv', index=False)

    out = tmp_path / 'out'
    out.mkdir()
    prov = {'version': '9.9.9-test', 'gitCommit': 'deadbee'}
    results = assembleMasterCsvs([str(proc)], {'P1': 'D1'}, str(out), provenance=prov)

    # master frame CSV produced with our three biomass rows
    assert (out / 'master_frame_features.csv').exists()
    assert results['frame'][1] == 3

    # provenance sidecar written and carries the injected pipeline record
    rec = json.loads((out / 'provenance.json').read_text())
    assert rec['pipeline'] == prov
    assert rec['masterCsvs']['frame']['rows'] == 3
    assert 'generatedAtUtc' in rec


# ---------------------------------------------------------------------------
# Registration sidecar: shifts + crop box rebuild `_registered_raw.tif` exactly
# from the raw frames, so the full stack need not be stored.
# ---------------------------------------------------------------------------

def test_registration_sidecar_rebuilds_registered_raw(tmp_path):
    import tifffile
    from multiWellAnalysis.processing.analysis_main import timelapseProcessing
    from multiWellAnalysis.processing.io_utils import readWellStack
    from multiWellAnalysis.processing.registration import (
        loadRegisteredRaw, loadRegistration, rebuildRegisteredRaw, registrationPath)

    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    T, H, W = 6, 160, 160
    # one fixed textured scene; the field of view drifts (2, 3) px per frame,
    # i.e. a real stage translation for phase correlation to recover
    scene = 40000 + 8000 * gaussian_filter(rng.normal(0, 1, (H + 40, W + 40)), 3)
    files = []
    for t in range(T):
        img = scene[2 * t:2 * t + H, 3 * t:3 * t + W]
        f = tmp_path / 'raw' / f'A1_03_1_1_Bright Field_{t + 1:03d}.tif'
        f.parent.mkdir(exist_ok=True)
        tifffile.imwrite(f, np.clip(img, 0, 65535).astype(np.uint16))
        files.append(str(f))

    out = tmp_path / 'out'
    timelapseProcessing(
        images=readWellStack(files), blockDiameter=31, ntimepoints=T,
        shiftThresh=50, fixedThresh=0.02, dustCorrection=True,
        outdir=str(out), filename='A1_03', imageRecords=None,
        fftStride=1, downsample=1, skipOverlay=True, saveRegistered=True,
        workers=1, sourceFiles=files,
    )
    procDir = out / 'processedImages'
    reg = loadRegistration(registrationPath(str(procDir), 'A1_03'))
    assert reg['shifts'].shape == (T, 2)
    # non-trivial shifts, so the rebuild exercises the warp (this is a round-trip
    # test of the sidecar, not of registration accuracy on a 160 px toy image)
    assert np.abs(reg['shifts'][1:]).max() > 1
    assert reg['sourceFiles'] == [os.path.abspath(f) for f in files]

    saved = np.transpose(tifffile.imread(procDir / 'A1_03_registered_raw.tif'), (1, 2, 0))
    rebuilt = rebuildRegisteredRaw(registrationPath(str(procDir), 'A1_03'))
    assert rebuilt.dtype == saved.dtype and rebuilt.shape == saved.shape
    assert np.array_equal(rebuilt, saved, equal_nan=True)

    # with the stack deleted, loadRegisteredRaw falls back to the rebuild
    os.remove(procDir / 'A1_03_registered_raw.tif')
    assert np.array_equal(loadRegisteredRaw(str(procDir), 'A1_03'), saved, equal_nan=True)


def test_limitThreads_caps_opencv():
    import cv2
    from multiWellAnalysis.processing.helpers import limitThreads
    before = cv2.getNumThreads()
    try:
        limitThreads()
        assert cv2.getNumThreads() == 1
    finally:
        cv2.setNumThreads(before)
