# -*- coding: utf-8 -*-
"""
FatigueAI 部署系统 v4 — 入口脚本

核心模块已拆分到 deploy/ 包:
    deploy/features.py  — 共享特征提取（唯一权威来源）
    deploy/train.py     — 模型训练（LOPO + 8模型 Stacking）
    deploy/monitor.py   — 实时监测（FatigueMonitor）
    deploy/demo.py      — 演示评估

直接运行本脚本 = 运行 demo()
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    from deploy.demo import demo
    demo()
