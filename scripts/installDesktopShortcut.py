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
    """Try multiple methods to find the conda base directory."""
    try:
        return subprocess.check_output(
            ['conda', 'info', '--base'], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass

    prefix = os.environ.get('CONDA_PREFIX', '')
    if prefix:
        candidate = os.path.dirname(os.path.dirname(prefix))
        condaSh = os.path.join(candidate, 'etc', 'profile.d', 'conda.sh')
        if os.path.isfile(condaSh):
            return candidate

    home = str(Path.home())
    for name in ['miniforge3', 'mambaforge', 'miniconda3', 'anaconda3',
                 'opt/miniconda3', 'opt/anaconda3']:
        candidate = os.path.join(home, name)
        if os.path.isfile(os.path.join(candidate, 'etc', 'profile.d', 'conda.sh')):
            return candidate

    return None


def _envNameFromBin(guiBin):
    """Extract the conda env name from a biofilm-processing-gui path, or None."""
    parts = Path(guiBin).resolve().parts
    try:
        idx = parts.index('envs')
        return parts[idx + 1]
    except (ValueError, IndexError):
        return None


def findGuiBin():
    """Find the biofilm-processing-gui executable, preferring named conda envs over base."""
    # Search named conda envs first (not base) — these are more likely correct
    condaBase = _findCondaBase()
    if condaBase:
        envsDir = os.path.join(condaBase, 'envs')
        if os.path.isdir(envsDir):
            for envName in sorted(os.listdir(envsDir)):
                if platform.system() == 'Windows':
                    candidate = os.path.join(envsDir, envName, 'Scripts', 'biofilm-processing-gui.exe')
                else:
                    candidate = os.path.join(envsDir, envName, 'bin', 'biofilm-processing-gui')
                if os.path.isfile(candidate):
                    return candidate

    # Check current env / PATH
    for envVar, subdir in [('CONDA_PREFIX', 'bin'), ('VIRTUAL_ENV', 'bin')]:
        prefix = os.environ.get(envVar)
        if prefix:
            if platform.system() == 'Windows':
                candidate = os.path.join(prefix, 'Scripts', 'biofilm-processing-gui.exe')
            else:
                candidate = os.path.join(prefix, subdir, 'biofilm-processing-gui')
            if os.path.isfile(candidate):
                return candidate

    gui = shutil.which('biofilm-processing-gui')
    if gui:
        return gui

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
            activateLines = f'# conda base not found, using absolute path\n'
    elif venv:
        activateLines = f'source "{venv}/bin/activate"\n'

    macosDir = os.path.join(desktopDir, 'biofilm-processing.app', 'Contents', 'MacOS')
    os.makedirs(macosDir, exist_ok=True)

    launcher = os.path.join(macosDir, 'biofilm-processing')
    with open(launcher, 'w') as f:
        f.write('#!/bin/zsh\n')
        f.write(f'# biofilm-processing GUI launcher — errors logged to {logPath}\n')
        f.write(f'exec >> "{logPath}" 2>&1\n')
        f.write(f'echo "--- $(date) ---"\n')
        f.write(f'echo "PATH=$PATH"\n\n')
        f.write(activateLines)
        f.write(f'\n# Try biofilm-processing-gui on PATH, then fall back to absolute path\n')
        f.write(f'if command -v biofilm-processing-gui &>/dev/null; then\n')
        f.write(f'    biofilm-processing-gui\n')
        f.write(f'else\n')
        f.write(f'    "{guiBinAbs}"\n')
        f.write(f'fi\n')
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
    condaPrefix = os.environ.get('CONDA_PREFIX')
    venv = os.environ.get('VIRTUAL_ENV')

    if condaPrefix and envName:
        activate = (
            f'call "{condaPrefix}\\Scripts\\activate.bat"\n'
            f'call conda activate {envName}\n'
        )
    elif venv:
        activate = f'call "{venv}\\Scripts\\activate.bat"\n'
    else:
        activate = ''

    # hide the .bat launcher in AppData so only the .lnk shows on Desktop
    appDataDir = os.path.join(os.environ.get('APPDATA', ''), 'biofilm-processing')
    os.makedirs(appDataDir, exist_ok=True)
    batPath = os.path.join(appDataDir, 'biofilm-processing.bat')
    with open(batPath, 'w') as f:
        f.write('@echo off\n')
        f.write(activate)
        f.write('biofilm-processing-gui\n')
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
