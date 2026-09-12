#!/usr/bin/env python3
"""Create a desktop shortcut for biofilm-processing GUI.

Works on Linux, macOS, and Windows.

Usage:
    python scripts/installDesktopShortcut.py
"""

import os
import sys
import shutil
import platform
import subprocess
import stat
from pathlib import Path


def _findCondaBase():
    """Locate the conda BASE install (the one holding Scripts/activate.bat).

    Deliberately tries several routes, because no single one is reliable:

    * `conda info --base` fails in PowerShell, where `conda` is a shell FUNCTION
      rather than an executable subprocess can spawn.
    * CONDA_EXE is set by every conda shell hook and points at the base's own
      conda executable, so it survives that -- the most dependable signal.
    * Deriving from CONDA_PREFIX only works when envs live inside the base. With
      an ALL-USERS install the base is under C:\ProgramData while envs land in
      %USERPROFILE%\.conda\envs (ProgramData is not user-writable), so
      stripping two components yields "<user>\.conda", which is an env store,
      not an install.
    * The directory scan must therefore include the system-wide locations, not
      just $HOME.

    Returns None if nothing is found; callers must not assume success.
    """
    try:
        return subprocess.check_output(
            ['conda', 'info', '--base'], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass

    # CONDA_EXE -> <base>/Scripts/conda.exe (Windows) or <base>/bin/conda
    condaExe = os.environ.get('CONDA_EXE', '')
    if condaExe:
        candidate = os.path.dirname(os.path.dirname(condaExe))
        if _looksLikeCondaBase(candidate):
            return candidate

    prefix = os.environ.get('CONDA_PREFIX', '')
    if prefix:
        if _looksLikeCondaBase(prefix):       # base env itself is active
            return prefix
        candidate = os.path.dirname(os.path.dirname(prefix))
        if _looksLikeCondaBase(candidate):
            return candidate

    home = str(Path.home())
    roots = [home, os.environ.get('LOCALAPPDATA', ''), os.environ.get('PROGRAMDATA', ''),
             'C:\\ProgramData', 'C:\\']
    names = ['miniforge3', 'mambaforge', 'miniconda3', 'anaconda3', 'Miniconda3',
             'Anaconda3', 'opt/miniconda3', 'opt/anaconda3']
    for root in [r for r in roots if r]:
        for name in names:
            candidate = os.path.join(root, name)
            if _looksLikeCondaBase(candidate):
                return candidate

    return None


def _looksLikeCondaBase(path):
    """True if `path` is a conda INSTALL rather than merely an env directory.

    Checks for the activation entry point we actually need on this platform;
    an env store such as %USERPROFILE%\.conda has neither.
    """
    if not path or not os.path.isdir(path):
        return False
    if platform.system() == 'Windows':
        return os.path.isfile(os.path.join(path, 'Scripts', 'activate.bat')) or \
               os.path.isfile(os.path.join(path, 'condabin', 'conda.bat'))
    return os.path.isfile(os.path.join(path, 'etc', 'profile.d', 'conda.sh'))


def _envNameFromBin(guiBin):
    """Extract the conda env name from a biofilm-processing-gui path, or None."""
    parts = Path(guiBin).resolve().parts
    try:
        idx = parts.index('envs')
        return parts[idx + 1]
    except (ValueError, IndexError):
        return None


def findGuiBin():
    """Find the biofilm-processing-gui executable.

    Order matters: the ACTIVE environment wins. A machine can legitimately hold
    several installs of this package at different versions -- notably
    biofilm-embeddings (uPULLI-DL) vendors this package as a pinned submodule and
    installs it into its own env, so `envs/biofilm-embeddings` contains an older
    pinned engine. Scanning `envs/` alphabetically (as this used to do first)
    therefore picked `biofilm-embeddings` over `biofilm-processing` and wired the
    shortcut to the pinned engine, which showed up as the GUI reporting a stale
    version. Whichever env the user ran this installer from is by far the best
    signal, so check that first and treat the directory scan as a last resort.
    """
    # 1. The env this installer is running in.
    for envVar in ('CONDA_PREFIX', 'VIRTUAL_ENV'):
        prefix = os.environ.get(envVar)
        if prefix:
            if platform.system() == 'Windows':
                candidate = os.path.join(prefix, 'Scripts', 'biofilm-processing-gui.exe')
            else:
                candidate = os.path.join(prefix, 'bin', 'biofilm-processing-gui')
            if os.path.isfile(candidate):
                return candidate

    # 2. PATH -- also reflects the active env.
    gui = shutil.which('biofilm-processing-gui')
    if gui:
        return gui

    # 3. Last resort: scan named envs. Ambiguous by nature (see above), so prefer
    # an env whose name matches this project before falling back to any match.
    condaBase = _findCondaBase()
    if condaBase:
        envsDir = os.path.join(condaBase, 'envs')
        if os.path.isdir(envsDir):
            names = sorted(os.listdir(envsDir))
            names.sort(key=lambda n: n != 'biofilm-processing')
            for envName in names:
                if platform.system() == 'Windows':
                    candidate = os.path.join(envsDir, envName, 'Scripts', 'biofilm-processing-gui.exe')
                else:
                    candidate = os.path.join(envsDir, envName, 'bin', 'biofilm-processing-gui')
                if os.path.isfile(candidate):
                    return candidate

    return None


def getDesktopDir():
    """Get the user's Desktop directory."""
    if platform.system() == 'Windows':
        return os.path.join(os.environ.get('USERPROFILE', ''), 'Desktop')
    return os.path.join(Path.home(), 'Desktop')


def getIconPath(fmt='png'):
    """Return path to the app icon in the requested format, or None."""
    repoDir = Path(__file__).resolve().parent.parent
    if fmt == 'icns':
        icon = repoDir / 'assets' / 'biofilm-processing-icon.icns'
        if icon.exists():
            return str(icon)
    if fmt == 'ico':
        icon = repoDir / 'assets' / 'biofilm-processing-icon.ico'
        if icon.exists():
            return str(icon)
    icon = repoDir / 'assets' / 'biofilm-processing-icon.png'
    if icon.exists():
        return str(icon)
    return None


def installLinux(guiBin):
    """Create .desktop file for Linux."""
    desktopDir = getDesktopDir()
    appDir = os.path.join(Path.home(), '.local', 'share', 'applications')

    envName = _envNameFromBin(guiBin) or os.environ.get('CONDA_DEFAULT_ENV')
    condaPrefix = os.environ.get('CONDA_PREFIX')
    venv = os.environ.get('VIRTUAL_ENV')

    if condaPrefix and envName:
        condaBase = _findCondaBase() or os.path.join(str(Path.home()), 'anaconda3')
        execLine = (
            f'bash -c \'source "{condaBase}/etc/profile.d/conda.sh" '
            f'&& conda activate {envName} && biofilm-processing-gui\''
        )
    elif venv:
        execLine = f'bash -c \'source "{venv}/bin/activate" && biofilm-processing-gui\''
    else:
        execLine = guiBin

    iconPath = getIconPath('png')
    iconLine = f'Icon={iconPath}\n' if iconPath else ''

    desktopEntry = (
        '[Desktop Entry]\n'
        'Name=biofilm-processing\n'
        'Comment=High-throughput biofilm phenotyping GUI\n'
        f'Exec={execLine}\n'
        'Terminal=false\n'
        'Type=Application\n'
        'Categories=Science;Education;\n'
        f'{iconLine}'
    )

    os.makedirs(appDir, exist_ok=True)
    appPath = os.path.join(appDir, 'biofilm-processing.desktop')
    with open(appPath, 'w') as f:
        f.write(desktopEntry)
    os.chmod(appPath, os.stat(appPath).st_mode | stat.S_IXUSR)
    print(f'Created: {appPath}')

    if os.path.isdir(desktopDir):
        deskPath = os.path.join(desktopDir, 'biofilm-processing.desktop')
        with open(deskPath, 'w') as f:
            f.write(desktopEntry)
        os.chmod(deskPath, os.stat(deskPath).st_mode | stat.S_IXUSR)
        print(f'Created: {deskPath}')

        if shutil.which('gio'):
            subprocess.run(
                ['gio', 'set', deskPath, 'metadata::trusted', 'true'],
                capture_output=True
            )


def installMacos(guiBin):
    """Create a .app bundle for macOS."""
    desktopDir = getDesktopDir()
    logPath = os.path.join(Path.home(), 'Library', 'Logs', 'biofilm-processing.log')

    envName = _envNameFromBin(guiBin) or os.environ.get('CONDA_DEFAULT_ENV')
    condaPrefix = os.environ.get('CONDA_PREFIX')
    venv = os.environ.get('VIRTUAL_ENV')
    guiBinAbs = os.path.realpath(guiBin)

    activateLines = ''
    if condaPrefix and envName:
        condaBase = _findCondaBase()
        if condaBase:
            activateLines = (
                f'# Activate conda environment\n'
                f'source "{condaBase}/etc/profile.d/conda.sh"\n'
                f'conda activate {envName}\n'
            )
        else:
            activateLines = '# conda base not found, using absolute path\n'
    elif venv:
        activateLines = f'source "{venv}/bin/activate"\n'

    macosDir = os.path.join(desktopDir, 'biofilm-processing.app', 'Contents', 'MacOS')
    os.makedirs(macosDir, exist_ok=True)

    launcher = os.path.join(macosDir, 'biofilm-processing')
    with open(launcher, 'w') as f:
        f.write('#!/bin/zsh\n')
        f.write(f'# biofilm-processing GUI launcher — errors logged to {logPath}\n')
        f.write(f'exec >> "{logPath}" 2>&1\n')
        f.write('echo "--- $(date) ---"\n')
        f.write('echo "PATH=$PATH"\n\n')
        f.write(activateLines)
        f.write('\n# Try biofilm-processing-gui on PATH, then fall back to absolute path\n')
        f.write('if command -v biofilm-processing-gui &>/dev/null; then\n')
        f.write('    biofilm-processing-gui\n')
        f.write('else\n')
        f.write(f'    "{guiBinAbs}"\n')
        f.write('fi\n')
    os.chmod(launcher, 0o755)

    contentsDir = os.path.join(desktopDir, 'biofilm-processing.app', 'Contents')
    with open(os.path.join(contentsDir, 'Info.plist'), 'w') as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0">\n'
            '<dict>\n'
            '  <key>CFBundleName</key>\n'
            '  <string>biofilm-processing</string>\n'
            '  <key>CFBundleExecutable</key>\n'
            '  <string>biofilm-processing</string>\n'
            '  <key>CFBundleIdentifier</key>\n'
            '  <string>edu.cmu.biofilm-processing</string>\n'
            '  <key>CFBundleVersion</key>\n'
            '  <string>0.1.0</string>\n'
            '  <key>CFBundleIconFile</key>\n'
            '  <string>biofilm-processing-icon</string>\n'
            '  <key>LSUIElement</key>\n'
            '  <false/>\n'
            '</dict>\n'
            '</plist>\n'
        )

    resDir = os.path.join(contentsDir, 'Resources')
    os.makedirs(resDir, exist_ok=True)
    icnsPath = getIconPath('icns')
    pngPath = getIconPath('png')
    if icnsPath:
        shutil.copy2(icnsPath, os.path.join(resDir, 'biofilm-processing-icon.icns'))
    if pngPath:
        shutil.copy2(pngPath, os.path.join(resDir, 'biofilm-processing-icon.png'))

    appPath = os.path.join(desktopDir, 'biofilm-processing.app')

    subprocess.run(
        ['xattr', '-dr', 'com.apple.quarantine', appPath],
        capture_output=True
    )

    print(f'Created: {appPath}')
    print(f'Errors will be logged to: {logPath}')
    print()
    print('If double-clicking does nothing, check the log above.')
    print('If macOS blocks it: right-click the app > Open > Open.')


def installWindows(guiBin):
    """Create a .bat launcher and a Start Menu shortcut for Windows."""
    desktopDir = getDesktopDir()

    envName = _envNameFromBin(guiBin) or os.environ.get('CONDA_DEFAULT_ENV')
    condaBase = _findCondaBase()
    venv = os.environ.get('VIRTUAL_ENV')

    # Launch the executable by ABSOLUTE PATH and put its env directories on PATH
    # ourselves, rather than relying on `conda activate` to do it.
    #
    # We already resolved guiBin, so activation was never load-bearing -- and it
    # is the fragile part. `_findCondaBase()` returns None whenever conda is not
    # discoverable: `conda info --base` is spawned with subprocess, but in
    # PowerShell `conda` is a shell FUNCTION, not an exe, so that call fails;
    # the CONDA_PREFIX fallback then derives the base by stripping two path
    # components, which gives "<user>\.conda" for the common
    # "<user>\.conda\envs\<name>" layout -- a directory that holds envs but is
    # not a conda install; and the home-directory scan only knows miniforge3 /
    # miniconda3 / anaconda3 under $HOME. With all three missing, the launcher
    # was written with NO activation line at all and failed with
    # "'biofilm-processing-gui' is not recognized".
    #
    # Scripts + Library\bin + the env root are what activation would have
    # prepended; including them keeps conda-provided DLLs (Qt, MKL) reachable.
    envDir = os.path.dirname(os.path.dirname(guiBin))   # ...\envs\<name>
    lines = [
        '@echo off\n',
        f'set "ENVDIR={envDir}"\n',
        'set "PATH=%ENVDIR%;%ENVDIR%\\Scripts;%ENVDIR%\\Library\\bin;%PATH%"\n',
    ]
    # Best-effort conda activation on top, when we could actually locate a base.
    # It adds the conda-managed environment variables some packages expect; the
    # absolute-path launch below does not depend on it succeeding.
    if condaBase and envName:
        lines.append(f'call "{condaBase}\\Scripts\\activate.bat" {envName} 2>nul\n')
    elif venv:
        lines.append(f'call "{venv}\\Scripts\\activate.bat" 2>nul\n')
    lines.append(f'"{guiBin}"\n')
    # Keep the window up on failure: the .lnk is created minimized, so without
    # this any startup error vanishes with the closing console.
    lines.append('if errorlevel 1 pause\n')

    # hide the .bat launcher in AppData so only the .lnk shows on Desktop
    appDataDir = os.path.join(os.environ.get('APPDATA', ''), 'biofilm-processing')
    os.makedirs(appDataDir, exist_ok=True)
    batPath = os.path.join(appDataDir, 'biofilm-processing.bat')
    with open(batPath, 'w') as f:
        f.writelines(lines)
    print(f'Created launcher: {batPath}')

    lnkPath = os.path.join(desktopDir, 'biofilm-processing.lnk')
    iconPath = getIconPath('ico')
    iconArg = f'$s.IconLocation = "{iconPath}"; ' if iconPath else ''
    psScript = (
        f'$ws = New-Object -ComObject WScript.Shell; '
        f'$s = $ws.CreateShortcut("{lnkPath}"); '
        f'$s.TargetPath = "{batPath}"; '
        f'$s.Description = "biofilm-processing - Biofilm Phenotyping GUI"; '
        f'{iconArg}'
        f'$s.WindowStyle = 7; '
        f'$s.Save()'
    )
    try:
        subprocess.run(
            ['powershell', '-Command', psScript],
            capture_output=True, check=True
        )
        print(f'Created shortcut: {lnkPath}')
    except Exception:
        # fallback: put .bat on Desktop if .lnk creation fails
        fallback = os.path.join(desktopDir, 'biofilm-processing.bat')
        shutil.copy2(batPath, fallback)
        print(f'Created: {fallback} (could not create .lnk shortcut)')


def main():
    guiBin = findGuiBin()
    if not guiBin:
        print('Error: biofilm-processing-gui not found.')
        print('Make sure you have run: pip install -e .')
        print('And that your conda/virtualenv is activated.')
        sys.exit(1)

    print(f'Found biofilm-processing-gui at: {guiBin}')
    envName = _envNameFromBin(guiBin)
    if envName:
        print(f'Detected conda env: {envName}')

    system = platform.system()
    if system == 'Linux':
        installLinux(guiBin)
    elif system == 'Darwin':
        installMacos(guiBin)
    elif system == 'Windows':
        installWindows(guiBin)
    else:
        print(f'Unsupported platform: {system}')
        sys.exit(1)

    print('\nbiofilm-processing shortcut installed.')
    print('You can launch it from your desktop or application menu.')


if __name__ == '__main__':
    main()
