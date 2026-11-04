# -*- mode: python ; coding: utf-8 -*-

# PyInstaller spec for the hsi-toolkit GUI (single-file executable).
# Build from the repository root:
#   uv run pyinstaller --noconfirm hsi-toolkit.spec
# The executable lands in dist/hsi-toolkit.exe.

a = Analysis(
    ['launcher.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    # mpl_toolkits.mplot3d is loaded dynamically via projection="3d".
    hiddenimports=['mpl_toolkits.mplot3d'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='hsi-toolkit.exe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
