# -*- coding: utf-8 -*-
"""FatigueAI 部署系统包

模块结构:
    deploy/features.py  — 共享特征提取函数（去EEG + 子窗口）
    deploy/train.py     — 模型训练 (Stacking + LOPO)
    deploy/monitor.py   — 实时监测类 (FatigueMonitor)
    deploy/demo.py      — 演示评估
"""

from .features import (
    extract_features,
    filter_eeg_features,
    normalize_by_baseline,
    slice_sensor_data,
    extract_with_subwindows,
)
from .monitor import FatigueMonitor
from .train import train_and_save
from .demo import demo
