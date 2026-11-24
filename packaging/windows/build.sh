#!/usr/bin/env bash
# Build the Windows installer from Linux: packaging/windows/build.sh
#
#   1. unpack the official embeddable Python for Windows,
#   2. install the locked dependencies (Windows wheels, fetched by uv) and the app into it,
#   3. package everything with NSIS into build/windows/BackingTrackGenerator-<version>-Setup.exe.
#
# Needs: uv, curl, unzip, sha256sum, and either a native `makensis` (AUR: nsis)
# or Wine (a portable NSIS is then downloaded and run under Wine, in its own prefix).
set -euo pipefail

PYTHON_VERSION=3.13.17
PYTHON_SHA256=9e35319bffedab8b36ad44d8f377cb4f1733bcec257a0b6d04d94f01486ddb38
NSIS_VERSION=3.13
NSIS_SHA256=ba63dffc4410ee89193e1cb5a41989991bd77c61068da17e3156d136b7b0b3d8
# Microsoft's C++ runtime DLLs (Microsoft-signed), repackaged on PyPI. Torch needs a newer
# runtime than Python and PySide6 ship (vcruntime140_threads.dll, msvcp140_atomic_wait.dll).
MSVC_WHEEL=msvc_runtime-14.51.36231-cp313-cp313-win_amd64.whl
MSVC_URL=https://files.pythonhosted.org/packages/c2/6e/7ba5696e9d8c1bcb01352f06e441497c308e84f75aba598400646ab31ca7/$MSVC_WHEEL
MSVC_SHA256=f3c3ac4f733a774f6ae209567e533bfde6f72dc1c10977733bfe09ce01e5b624
TORCH_INDEX=https://download.pytorch.org/whl/cpu

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=$ROOT/build/windows
DL=$OUT/downloads
STAGE=$OUT/stage
PY=$STAGE/python
SITE=$PY/Lib/site-packages

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)
SETUP=$OUT/BackingTrackGenerator-$VERSION-Setup.exe

log() { printf '\033[1m==> %s\033[0m\n' "$*"; }

# download URL FILE SHA256: fetch once into $DL, verify the checksum every time.
download() {
    local url=$1 file=$DL/$2 sha=$3
    if [[ ! -f $file ]]; then
        log "Downloading $2"
        curl -fsSL -o "$file.part" "$url"
        mv "$file.part" "$file"
    fi
    echo "$sha  $file" | sha256sum -c --quiet - || { rm -f "$file"; echo "Checksum mismatch: $2" >&2; exit 1; }
}

mkdir -p "$DL"
rm -rf "$STAGE"
mkdir -p "$PY"

# ----- Python ----------------------------------------------------------------
PY_ZIP=python-$PYTHON_VERSION-embed-amd64.zip
download "https://www.python.org/ftp/python/$PYTHON_VERSION/$PY_ZIP" "$PY_ZIP" "$PYTHON_SHA256"
log "Unpacking Python $PYTHON_VERSION"
unzip -q "$DL/$PY_ZIP" -d "$PY"
# The embeddable Python ignores PYTHONPATH and site-packages unless its ._pth says otherwise.
PTH=$(ls "$PY"/python3*._pth)
STDLIB_ZIP=$(basename "${PTH%._pth}").zip
printf '%s\r\n.\r\nLib\\site-packages\r\nimport site\r\n' "$STDLIB_ZIP" > "$PTH"

# ----- dependencies and the app ------------------------------------------------
log "Installing dependencies (Windows wheels)"
# The exact versions from uv.lock. Only Qt's essentials: the app uses QtCore/QtGui/QtWidgets,
# and the full PySide6 adds WebEngine, Quick 3D, ... (several hundred MB).
uv export --project "$ROOT" --frozen --no-dev --no-hashes --no-emit-project --no-header \
    | grep -vE '^pyside6(-addons|-pdf|-webengine)?==' > "$OUT/requirements.txt"
PY_MINOR=${PYTHON_VERSION%.*}
pip_install() {
    uv pip install --quiet --no-deps --target "$SITE" \
        --python-platform x86_64-pc-windows-msvc --python-version "$PY_MINOR" \
        --only-binary :all: "$@"
}
pip_install --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match -r "$OUT/requirements.txt"

log "Building the app wheel"
rm -rf "$OUT/wheel"
uv build --project "$ROOT" --wheel --quiet -o "$OUT/wheel"
pip_install "$OUT"/wheel/*.whl

log "Trimming files not needed at run time"
rm -rf "$SITE/torch/include" "$SITE/torch/share" "$SITE"/torch/lib/*.lib
rm -rf "$SITE"/PySide6/{qml,metatypes,typesystems,glue,include,scripts} \
       "$SITE"/PySide6/*.exe "$SITE"/PySide6/Qt6Designer*.dll "$SITE"/PySide6/plugins/designer
rm -rf "$SITE/bin"  # console scripts made for Linux: not usable here
# The C++ runtime goes next to python.exe, where Windows finds it for every package
# (app-local deployment): no VC++ redistributable, and no administrator rights, needed.
download "$MSVC_URL" "$MSVC_WHEEL" "$MSVC_SHA256"
log "Adding the Microsoft C++ runtime"
unzip -q -o -j "$DL/$MSVC_WHEEL" '*/data/Scripts/vcruntime140*.dll' '*/data/Scripts/msvcp140*.dll' \
    '*/data/Scripts/concrt140.dll' '*/data/Scripts/vcomp140.dll' -d "$PY"

# ----- installer -----------------------------------------------------------------
log "Writing the icon"
uv run --project "$ROOT" --quiet python "$HERE/make_icon.py" "$STAGE/app.ico"
cp "$ROOT/LICENSE" "$STAGE/LICENSE.txt"

log "Running NSIS"
NSIS_DEFINES=(-V2 "-DVERSION=$VERSION")
if command -v makensis >/dev/null; then
    makensis "${NSIS_DEFINES[@]}" "-DSTAGE=$STAGE" "-DOUTFILE=$SETUP" "$HERE/installer.nsi"
else
    command -v wine >/dev/null || { echo "Need makensis or wine" >&2; exit 1; }
    NSIS_ZIP=nsis-$NSIS_VERSION.zip
    download "https://downloads.sourceforge.net/project/nsis/NSIS%203/$NSIS_VERSION/$NSIS_ZIP" "$NSIS_ZIP" "$NSIS_SHA256"
    [[ -d $OUT/nsis-$NSIS_VERSION ]] || unzip -q "$DL/$NSIS_ZIP" -d "$OUT"
    export WINEPREFIX=$OUT/wineprefix WINEDEBUG=-all WINEDLLOVERRIDES="mscoree,mshtml=;winemenubuilder.exe=d"
    winpath() { wine winepath -w "$1" 2>/dev/null | tr -d '\r'; }
    wine "$OUT/nsis-$NSIS_VERSION/makensis.exe" "${NSIS_DEFINES[@]}" \
        "-DSTAGE=$(winpath "$STAGE")" "-DOUTFILE=$(winpath "$SETUP")" "$(winpath "$HERE/installer.nsi")"
fi

log "Done: $SETUP ($(du -h "$SETUP" | cut -f1))"
