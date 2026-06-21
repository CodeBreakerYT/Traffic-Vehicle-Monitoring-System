# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('D:\\Github\\Traffic-Vehicle-Monitoring-System\\assets\\icon', 'assets/icon'), ('D:\\Github\\Traffic-Vehicle-Monitoring-System\\assets\\gif', 'assets/gif'), ('D:\\Github\\Traffic-Vehicle-Monitoring-System\\assets\\training\\traffic_dataset\\model\\accident_detector.pt', 'assets/training/traffic_dataset/model'), ('D:\\Github\\Traffic-Vehicle-Monitoring-System\\assets\\training\\traffic_dataset\\model\\severity_detector.pt', 'assets/training/traffic_dataset/model'), ('D:\\Github\\Traffic-Vehicle-Monitoring-System\\assets\\training\\simulation_dataset\\simulation_detector.pt', 'assets/training/simulation_dataset'), ('D:\\Github\\Traffic-Vehicle-Monitoring-System\\assets\\training\\simulation_dataset\\simulation_accident_detector.pt', 'assets/training/simulation_dataset'), ('D:\\Github\\Traffic-Vehicle-Monitoring-System\\config.json', '.')]
binaries = []
hiddenimports = ['scripts.option_3.simulation_client']
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='car',
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
    icon=['D:\\Github\\Traffic-Vehicle-Monitoring-System\\assets\\icon\\car.ico'],
)
