import os

def test_imports():
    pass


def test_headless_cli_buildState(monkeypatch):
    # The headless runner must import without a display and merge config <
    # CLI overrides, preserving outputDir/plates (not in state DEFAULTS).
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from multiWellAnalysis.cli.run_pipeline import buildState
    state = buildState(None, {'plates': ['/data/p'], 'outputDir': '/out',
                              'workers': 40, 'colonyTracking': None})
    assert state['plates'] == ['/data/p']
    assert state['outputDir'] == '/out'
    assert state['workers'] == 40
    assert state['colonyTracking'] is False  # None override -> DEFAULTS value


def test_headless_test_well_parser(monkeypatch):
    # Single-well verification CLI must import headless and parse plate/well args.
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from multiWellAnalysis.cli.test_well import buildParser
    args = buildParser().parse_args(['/data/plate', '--well', 'B2', '--mag', '_03',
                                     '--fixed-thresh', '0.03', '--tracking'])
    assert args.plate == '/data/plate'
    assert args.well == 'B2'
    assert args.mag == '_03'
    assert args.fixedThresh == 0.03
    assert args.tracking is True


def test_save_registered_flag(monkeypatch):
    # `_registered_raw.tif` is the largest per-well output and biomass does not
    # need it, so --no-save-registered must reach the worker state and the
    # per-well kwarg. Defaults stay True: skipping is opt-in.
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    import inspect
    from multiWellAnalysis.cli.run_pipeline import buildParser, buildState
    from multiWellAnalysis.processing.analysis_main import timelapseProcessing

    assert buildState(None, {})['saveRegistered'] is True
    args = buildParser().parse_args(['--output-dir', '/out', '--plates', '/d',
                                     '--no-save-registered'])
    assert args.saveRegistered is False
    assert buildState(None, {'saveRegistered': args.saveRegistered})[
        'saveRegistered'] is False

    sig = inspect.signature(timelapseProcessing)
    assert sig.parameters['saveRegistered'].default is True


def test_save_registered_refuses_colony_stages(monkeypatch, capsys):
    # Tracking and colony features read the raw stack. Skipping the write while
    # asking for them must fail immediately, not ten hours in.
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from multiWellAnalysis.cli.run_pipeline import main
    rc = main(['--output-dir', '/out', '--plates', '/d',
               '--no-save-registered', '--colony-tracking'])
    assert rc == 2
    assert 'registered_raw' in capsys.readouterr().err


def test_saveStack_trims_speculative_prealloc(tmp_path):
    # XFS answers tifffile's incremental page writes with post-EOF speculative
    # preallocation that is never trimmed on close (measured 2.37x on a 25-page
    # float32 stack, persisting for hours), which would inflate a 96-well plate
    # from 35 GiB to 83 GiB. saveStack must release it, without touching content.
    import numpy as np
    import tifffile
    from multiWellAnalysis.processing.io_utils import saveStack

    stack = np.random.default_rng(0).random((16, 17, 3)).astype(np.float32)
    saveStack(stack, str(tmp_path), 'w1')
    path = tmp_path / 'w1.tif'

    st = os.stat(path)
    allocated = getattr(st, 'st_blocks', 0) * 512
    # Never over-allocated after the trim (filesystems without post-EOF
    # preallocation already satisfy this; the point is that we never exceed it).
    assert allocated <= max(st.st_size, 4096) * 2
    # Content survives the truncate: (H, W, T) in -> (T, H, W) on disk.
    assert np.array_equal(tifffile.imread(str(path)),
                          np.transpose(stack, (2, 0, 1)))


def test_nas_rsync_flags_are_cifs_safe(monkeypatch):
    # Regression: `rsync -a` cannot write to CIFS/SMB (forced uid/gid/mode reject
    # chown/chgrp/chmod and even the perms-preserving temp-file mkstemp), so the
    # NAS mirror silently transferred nothing. The flags must preserve no
    # attributes so the copy actually lands.
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from multiWellAnalysis.gui.tabs.run import _NAS_RSYNC
    assert _NAS_RSYNC[0] == 'rsync'
    assert '-a' not in _NAS_RSYNC
    for flag in ('--no-perms', '--no-owner', '--no-group', '--no-times'):
        assert flag in _NAS_RSYNC, f'{flag} missing from _NAS_RSYNC'
