#!/bin/bash
# Create a desktop shortcut for Phenotypr GUI.
# Works on Linux with any freedesktop.org-compatible desktop (GNOME, KDE, XFCE, etc.)
#
# Usage:
#   bash scripts/install-desktop-shortcut.sh
#
# This will:
#   1. Find the phenotypr-gui entry point installed by pip
#   2. Create a .desktop file on the user's Desktop and in ~/.local/share/applications/
#   3. Optionally copy an icon if one exists

set -e

DESKTOP_DIR="${HOME}/Desktop"
APP_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons"

# Find the phenotypr-gui executable
GUI_BIN=$(command -v phenotypr-gui 2>/dev/null || true)

if [ -z "$GUI_BIN" ]; then
    # Try conda/venv bin
    if [ -n "$CONDA_PREFIX" ]; then
        GUI_BIN="$CONDA_PREFIX/bin/phenotypr-gui"
    elif [ -n "$VIRTUAL_ENV" ]; then
        GUI_BIN="$VIRTUAL_ENV/bin/phenotypr-gui"
    fi
fi

if [ -z "$GUI_BIN" ] || [ ! -f "$GUI_BIN" ]; then
    echo "Error: phenotypr-gui not found."
    echo "Make sure you've run: pip install -e ."
    echo "And that your conda/virtualenv is activated."
    exit 1
fi

echo "Found phenotypr-gui at: $GUI_BIN"

# Determine the conda/venv activation needed
EXEC_LINE="$GUI_BIN"
if [ -n "$CONDA_PREFIX" ] && [ -n "$CONDA_DEFAULT_ENV" ]; then
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
    EXEC_LINE="bash -c 'source \"$CONDA_BASE/etc/profile.d/conda.sh\" && conda activate $CONDA_DEFAULT_ENV && phenotypr-gui'"
    echo "Will activate conda env: $CONDA_DEFAULT_ENV"
elif [ -n "$VIRTUAL_ENV" ]; then
    EXEC_LINE="bash -c 'source \"$VIRTUAL_ENV/bin/activate\" && phenotypr-gui'"
    echo "Will activate venv: $VIRTUAL_ENV"
fi

# Check for icon
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
ICON_SRC="$REPO_DIR/assets/phenotypr-icon.png"
ICON_PATH=""

if [ -f "$ICON_SRC" ]; then
    mkdir -p "$ICON_DIR"
    cp "$ICON_SRC" "$ICON_DIR/phenotypr.png"
    ICON_PATH="$ICON_DIR/phenotypr.png"
    echo "Installed icon to: $ICON_PATH"
fi

# Write .desktop file
write_desktop_file() {
    local dest="$1"
    cat > "$dest" <<DESKTOP
[Desktop Entry]
Name=Phenotypr
Comment=High-throughput biofilm phenotyping GUI
Exec=$EXEC_LINE
Terminal=false
Type=Application
Categories=Science;Education;
${ICON_PATH:+Icon=$ICON_PATH}
DESKTOP
    chmod +x "$dest"
    echo "Created: $dest"
}

# Install to applications menu
mkdir -p "$APP_DIR"
write_desktop_file "$APP_DIR/phenotypr.desktop"

# Install to Desktop if it exists
if [ -d "$DESKTOP_DIR" ]; then
    write_desktop_file "$DESKTOP_DIR/phenotypr.desktop"
    # Mark as trusted on GNOME (Ubuntu)
    if command -v gio &>/dev/null; then
        gio set "$DESKTOP_DIR/phenotypr.desktop" metadata::trusted true 2>/dev/null || true
    fi
fi

echo ""
echo "Phenotypr shortcut installed."
echo "You can launch it from your desktop or application menu."
