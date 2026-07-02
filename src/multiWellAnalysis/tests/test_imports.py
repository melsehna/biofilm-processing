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
