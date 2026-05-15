# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['qt/train_gui.py'],
    pathex=['.'],             # 关键：让 PyInstaller 能找到 lib/ 和 modules/
    binaries=[],
    datas=[],
    hiddenimports=[
        # TensorFlow 全家桶
        'tensorflow',
        'tensorflow.python',
        'tensorflow.python.keras',
        'tensorflow_model_optimization',
        # PyQt5
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        # matplotlib + seaborn
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'seaborn',
        # opencv / sklearn / tqdm
        'cv2',
        'sklearn',
        'tqdm',
        # 项目自身模块（动态导入的也要写上）
        'lib',
        'lib.AU',
        'lib.polt_improved',
        'lib.show_img',
        'modules',
        'modules.model_utils',
        'modules.export_utils',
        'modules.plot_utils',
        'modules.model_test_utils',
    ],
    hookspath=[],
    hooksconfig={},
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
    name='train_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,              # 不显示命令行黑窗
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='train_gui',           # 输出文件夹名称
)