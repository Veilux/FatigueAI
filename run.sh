#!/bin/bash
# ==============================================================
#  FatigueAI 比赛一键启动脚本
#  用法: bash run.sh [train|web|demo|all]
# ==============================================================
set -e
cd "$(dirname "$0")"

PYTHON=python3
PORT=8080

echo "============================================"
echo "  FatigueAI — 运动疲劳智能监测系统"
echo "============================================"

case "${1:-all}" in
  check)
    echo "[1/1] 检查数据集..."
    $PYTHON main.py --check
    ;;
  train)
    echo "[1/1] 训练模型 (v4 Stacking)..."
    $PYTHON fatigue_deploy_v4.py
    echo "模型已保存到 outputs/models/deploy_v4_model.pkl"
    ;;
  web)
    echo "[启动] Web 服务 http://localhost:$PORT"
    $PYTHON -m web.server
    ;;
  demo)
    echo "[1/2] 检查数据集..."
    $PYTHON main.py --check
    echo "[2/2] 运行 Demo (训练 + 评估)..."
    $PYTHON fatigue_deploy_v4.py
    ;;
  all)
    echo "[1/3] 检查数据集..."
    $PYTHON main.py --check
    echo ""
    echo "[2/3] 训练模型..."
    $PYTHON fatigue_deploy_v4.py
    echo ""
    echo "[3/3] 启动 Web 服务..."
    echo "  访问 http://localhost:$PORT"
    $PYTHON -m web.server
    ;;
  *)
    echo "用法: bash run.sh [check|train|web|demo|all]"
    echo "  check  - 仅检查数据集"
    echo "  train  - 训练模型并保存"
    echo "  web    - 启动 Web 服务"
    echo "  demo   - 训练 + Demo 评估"
    echo "  all    - 完整流程: 检查 → 训练 → Web (默认)"
    ;;
esac
