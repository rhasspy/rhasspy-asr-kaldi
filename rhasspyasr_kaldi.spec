# -*- mode: python -*-
import os
import site
from pathlib import Path

block_cipher = None

a = Analysis(
    [Path.cwd() / "__main__.py"],
    pathex=["."],
    binaries=[
        ("rhasspyasr_kaldi/libfst.so.13", "."),
        ("rhasspyasr_kaldi/libfstfar.so.13", "."),
        ("rhasspyasr_kaldi/libfstngram.so.13", "."),
        ("rhasspyasr_kaldi/phonetisaurus-g2pfst", "."),
    ],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="rhasspyasr_kaldi",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, name="rhasspyasr_kaldi"
)
