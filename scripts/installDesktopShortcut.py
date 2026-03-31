#!/usr/bin/env python3
"""Create a desktop shortcut for Phenotypr GUI.

Works on Linux, macOS, and Windows.

Usage:
    python scripts/install-desktop-shortcut.py
"""

import os
import sys
import shutil
import platform
import subprocess
import stat
from pathlib import Path


def find_gui_bin():
    """Find the phenotypr-gui executable."""
    gui = shutil.which('phenotypr-gui')
    if gui:
        return gui

    # Check conda/venv bin directories
    for env_var, subdir in [('CONDA_PREFIX', 'bin'), ('VIRTUAL_ENV', 'bin')]:
        prefix = os.environ.get(env_var)
        if prefix:
            if platform.system() == 'Windows':
                candidate = os.path.join(prefix, 'Scripts', 'phenotypr-gui.exe')
            else:
                candidate = os.path.join(prefix, subdir, 'phenotypr-gui')
            if os.path.isfile(candidate):
                return candidate

    return None


def find_python():
    """Find the current python executable path."""
    return sys.executable


def get_desktop_dir():
    """Get the user's Desktop directory."""
    if platform.system() == 'Windows':
        return os.path.join(os.environ.get('USERPROFILE', ''), 'Desktop')
    elif platform.system() == 'Darwin':
        return os.path.join(Path.home(), 'Desktop')
    else:
        return os.path.join(Path.home(), 'Desktop')


def getIconPath(fmt='png'):
    """Return path to the app icon in the requested format, or None."""
    repoDir = Path(__file__).resolve().parent.parent
    if fmt == 'icns':
        icon = repoDir / 'assets' / 'phenotypr-icon.icns'
        if icon.exists():
            return str(icon)
    icon = repoDir / 'assets' / 'phenotypr-icon.png'
    if icon.exists():
        return str(icon)
    return None


def install_linux(gui_bin):
    """Create .desktop file for Linux."""
    desktop_dir = get_desktop_dir()
    app_dir = os.path.join(Path.home(), '.local', 'share', 'applications')

    # Build exec line with conda/venv activation
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    venv = os.environ.get('VIRTUAL_ENV')

    if conda_prefix and conda_env:
        try:
            conda_base = subprocess.check_output(
                ['conda', 'info', '--base'], text=True
            ).strip()
        except Exception:
            conda_base = os.path.join(str(Path.home()), 'anaconda3')
        exec_line = (
            f'bash -c \'source "{conda_base}/etc/profile.d/conda.sh" '
            f'&& conda activate {conda_env} && phenotypr-gui\''
        )
    elif venv:
        exec_line = f'bash -c \'source "{venv}/bin/activate" && phenotypr-gui\''
    else:
        exec_line = gui_bin

    iconPath = getIconPath('png')
    iconLine = f'Icon={iconPath}\n' if iconPath else ''

    desktop_entry = (
        '[Desktop Entry]\n'
        'Name=Phenotypr\n'
        'Comment=High-throughput biofilm phenotyping GUI\n'
        f'Exec={exec_line}\n'
        'Terminal=false\n'
        'Type=Application\n'
        'Categories=Science;Education;\n'
        f'{iconLine}'
    )

    # Install to application menu
    os.makedirs(app_dir, exist_ok=True)
    app_path = os.path.join(app_dir, 'phenotypr.desktop')
    with open(app_path, 'w') as f:
        f.write(desktop_entry)
    os.chmod(app_path, os.stat(app_path).st_mode | stat.S_IXUSR)
    print(f'Created: {app_path}')

    # Install to Desktop
    if os.path.isdir(desktop_dir):
        desk_path = os.path.join(desktop_dir, 'phenotypr.desktop')
        with open(desk_path, 'w') as f:
            f.write(desktop_entry)
        os.chmod(desk_path, os.stat(desk_path).st_mode | stat.S_IXUSR)
        print(f'Created: {desk_path}')

        # Mark as trusted on GNOME
        if shutil.which('gio'):
            subprocess.run(
                ['gio', 'set', desk_path, 'metadata::trusted', 'true'],
                capture_output=True
            )


def _find_conda_base():
    """Try multiple methods to find the conda base directory."""
    # Method 1: conda info --base
    try:
        return subprocess.check_output(
            ['conda', 'info', '--base'], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass

    # Method 2: walk up from CONDA_PREFIX
    prefix = os.environ.get('CONDA_PREFIX', '')
    if prefix:
        # CONDA_PREFIX is e.g. ~/miniforge3/envs/phenotypr
        # base is 2 levels up if envs/ is in the path
        candidate = os.path.dirname(os.path.dirname(prefix))
        conda_sh = os.path.join(candidate, 'etc', 'profile.d', 'conda.sh')
        if os.path.isfile(conda_sh):
            return candidate

    # Method 3: check common install locations
    home = str(Path.home())
    for name in ['miniforge3', 'mambaforge', 'miniconda3', 'anaconda3',
                 'opt/miniconda3', 'opt/anaconda3']:
        candidate = os.path.join(home, name)
        if os.path.isfile(os.path.join(candidate, 'etc', 'profile.d', 'conda.sh')):
            return candidate

    return None


def install_macos(gui_bin):
    """Create a .app bundle for macOS.

    Key macOS issues this handles:
    - Finder launches .app with a minimal PATH (no conda/brew/etc.)
    - Default shell is zsh since Catalina, but conda.sh works in both
    - Quarantine attribute blocks unsigned .app bundles
    - Errors are logged to ~/Library/Logs/Phenotypr.log for debugging
    """
    desktop_dir = get_desktop_dir()
    log_path = os.path.join(Path.home(), 'Library', 'Logs', 'Phenotypr.log')

    # Build activation command
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    venv = os.environ.get('VIRTUAL_ENV')

    # Also record the full absolute path to phenotypr-gui as fallback
    gui_bin_abs = os.path.realpath(gui_bin)

    activate_lines = ''
    if conda_prefix and conda_env:
        conda_base = _find_conda_base()
        if conda_base:
            activate_lines = (
                f'# Activate conda environment\n'
                f'source "{conda_base}/etc/profile.d/conda.sh"\n'
                f'conda activate {conda_env}\n'
            )
        else:
            # Fallback: use the absolute path directly
            activate_lines = f'# conda base not found, using absolute path\n'
    elif venv:
        activate_lines = f'source "{venv}/bin/activate"\n'

    # Create .app bundle
    app_dir = os.path.join(desktop_dir, 'Phenotypr.app', 'Contents', 'MacOS')
    os.makedirs(app_dir, exist_ok=True)

    launcher = os.path.join(app_dir, 'phenotypr')
    with open(launcher, 'w') as f:
        f.write('#!/bin/zsh\n')
        f.write(f'# Phenotypr GUI launcher — errors logged to {log_path}\n')
        f.write(f'exec >> "{log_path}" 2>&1\n')
        f.write(f'echo "--- $(date) ---"\n')
        f.write(f'echo "PATH=$PATH"\n\n')
        f.write(activate_lines)
        f.write(f'\n# Try phenotypr-gui on PATH, then fall back to absolute path\n')
        f.write(f'if command -v phenotypr-gui &>/dev/null; then\n')
        f.write(f'    phenotypr-gui\n')
        f.write(f'else\n')
        f.write(f'    "{gui_bin_abs}"\n')
        f.write(f'fi\n')
    os.chmod(launcher, 0o755)

    # Write Info.plist with icon reference
    plistDir = os.path.join(desktop_dir, 'Phenotypr.app', 'Contents')
    with open(os.path.join(plistDir, 'Info.plist'), 'w') as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0">\n'
            '<dict>\n'
            '  <key>CFBundleName</key>\n'
            '  <string>Phenotypr</string>\n'
            '  <key>CFBundleExecutable</key>\n'
            '  <string>phenotypr</string>\n'
            '  <key>CFBundleIdentifier</key>\n'
            '  <string>edu.cmu.phenotypr</string>\n'
            '  <key>CFBundleVersion</key>\n'
            '  <string>0.1.0</string>\n'
            '  <key>CFBundleIconFile</key>\n'
            '  <string>phenotypr-icon</string>\n'
            '  <key>LSUIElement</key>\n'
            '  <false/>\n'
            '</dict>\n'
            '</plist>\n'
        )

    # Copy icon files (.icns for Finder, .png as fallback)
    icnsPath = getIconPath('icns')
    pngPath = getIconPath('png')
    resDir = os.path.join(plistDir, 'Resources')
    os.makedirs(resDir, exist_ok=True)
    if icnsPath:
        shutil.copy2(icnsPath, os.path.join(resDir, 'phenotypr-icon.icns'))
    if pngPath:
        shutil.copy2(pngPath, os.path.join(resDir, 'phenotypr-icon.png'))

    app_path = os.path.join(desktop_dir, 'Phenotypr.app')

    # Remove quarantine attribute so macOS doesn't block it
    subprocess.run(
        ['xattr', '-dr', 'com.apple.quarantine', app_path],
        capture_output=True
    )

    print(f'Created: {app_path}')
    print(f'Errors will be logged to: {log_path}')
    print()
    print('If double-clicking does nothing, check the log above.')
    print('If macOS blocks it: right-click the app > Open > Open.')


def install_windows(gui_bin):
    """Create a .bat launcher and a Start Menu shortcut for Windows."""
    desktop_dir = get_desktop_dir()
    python_exe = find_python()

    # Build activation command
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    venv = os.environ.get('VIRTUAL_ENV')

    if conda_prefix and conda_env:
        activate = (
            f'call "{conda_prefix}\\Scripts\\activate.bat"\n'
            f'call conda activate {conda_env}\n'
        )
    elif venv:
        activate = f'call "{venv}\\Scripts\\activate.bat"\n'
    else:
        activate = ''

    # Create .bat launcher
    bat_path = os.path.join(desktop_dir, 'Phenotypr.bat')
    with open(bat_path, 'w') as f:
        f.write('@echo off\n')
        f.write(activate)
        f.write('phenotypr-gui\n')
    print(f'Created: {bat_path}')

    # Try to create a proper .lnk shortcut via PowerShell
    lnk_path = os.path.join(desktop_dir, 'Phenotypr.lnk')
    iconPath = getIconPath('png')
    iconArg = f'$s.IconLocation = "{iconPath}"; ' if iconPath else ''
    ps_script = (
        f'$ws = New-Object -ComObject WScript.Shell; '
        f'$s = $ws.CreateShortcut("{lnk_path}"); '
        f'$s.TargetPath = "{bat_path}"; '
        f'$s.Description = "Phenotypr - Biofilm Phenotyping GUI"; '
        f'{iconArg}'
        f'$s.WindowStyle = 7; '  # minimized (hides the cmd window faster)
        f'$s.Save()'
    )
    try:
        subprocess.run(
            ['powershell', '-Command', ps_script],
            capture_output=True, check=True
        )
        print(f'Created: {lnk_path}')
    except Exception:
        print(f'Note: Could not create .lnk shortcut. Use {bat_path} to launch.')


def main():
    gui_bin = find_gui_bin()
    if not gui_bin:
        print('Error: phenotypr-gui not found.')
        print('Make sure you have run: pip install -e .')
        print('And that your conda/virtualenv is activated.')
        sys.exit(1)

    print(f'Found phenotypr-gui at: {gui_bin}')

    system = platform.system()
    if system == 'Linux':
        install_linux(gui_bin)
    elif system == 'Darwin':
        install_macos(gui_bin)
    elif system == 'Windows':
        install_windows(gui_bin)
    else:
        print(f'Unsupported platform: {system}')
        sys.exit(1)

    print('\nPhenotypr shortcut installed.')
    print('You can launch it from your desktop or application menu.')


if __name__ == '__main__':
    main()
