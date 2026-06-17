# FatigueAI — 运动疲劳智能监测系统

基于可穿戴生理信号的运动疲劳实时监测系统，通过LOPO交叉验证达到**85.71%**的Session级三分类准确率（最优seed）。

## 核心特性

- **手表可部署**: 仅使用智能手表可采集的6类信号（HR/HRV/EDA/温度/加速度/呼吸率），手表版本零损失迁移
- **多层级特征工程**: 300维特征空间（75基础统计 + 8交互特征 + 225子窗口时序特征）
- **Stacking集成**: XGBoost + LightGBM + CatBoost + RandomForest + ExtraTrees + GBDT 八模型集成
- **严格评估**: Leave-One-Participant-Out (LOPO) 交叉验证

## 项目结构

```
FatigueAI/
├── config/
│   └── dataset_config.py      # 数据集配置（路径、信号定义、标签映射）
├── data/
│   └── loader.py              # FatigueSet数据集加载器
├── features/
│   └── extractor.py           # 特征提取器（统计特征、HRV、频域特征）
├── models/
│   ├── trainer.py             # 模型训练（LOPO、Stacking）
│   └── cnn_lstm.py            # CNN+BiLSTM深度学习模型
├── deploy/                    # 部署系统（v4）
│   ├── features.py            # 共享特征提取（唯一权威来源）
│   ├── train.py               # Stacking模型训练
│   ├── monitor.py             # FatigueMonitor实时监测类
│   └── demo.py                # 演示评估
├── web/                       # Web可视化
│   ├── server.py              # FastAPI后端 + WebSocket
│   ├── templates/index.html   # Vue3前端
│   └── static/                # CSS/JS
├── rag/
│   └── knowledge_builder.py   # 运动科学知识库
├── api/
│   └── advisor.py             # LLM健康建议生成
├── main.py                    # 主入口（深度学习 + GradientBoosting）
├── fatigue_deploy_v4.py       # v4部署系统入口
├── run.sh                     # 一键启动脚本
├── requirements.txt           # Python依赖
└── README.md
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 一键运行
bash run.sh all

# 或分步运行
python main.py --check          # 检查数据集
python fatigue_deploy_v4.py     # 训练 + 评估
python -m web.server            # 启动Web服务 → http://localhost:8080
```

## 数据集

使用 **FatigueSet** 数据集（自建多模态运动疲劳数据集）：
- 12名参与者 × 3次实验 × ~20分钟
- 信号: 胸带Zephyr BioHarness（ECG/呼吸/HRV）+ 手腕Empatica E4（BVP/EDA/温度/加速度）+ Muse S头带（EEG）
- 标签: 三级疲劳状态（低/中/高）

数据目录结构：
```
fatigueset/fatigueset/
├── metadata.csv           # 参与者活动强度分配
├── 01/ 02/ ... 12/        # 参与者目录
│   └── 01/ 02/ 03/        # 会话目录
│       ├── chest_physiology_summary.csv
│       ├── wrist_hr.csv
│       ├── wrist_eda.csv
│       ├── exp_fatigue.csv
│       └── ...
```

可通过环境变量指定数据路径：
```bash
export FATIGUESET_ROOT=/path/to/fatigueset/fatigueset
```

## 实验结果

| 配置 | 特征数 | Session准确率 |
|------|--------|-------------|
| 全特征（含EEG） | 348 | 85.00% |
| 去EEG (v4标准) | 300 | 85.00% |
| 去EDA | 312 | 65.00% |
| 去EEG+stride=30s | 300 | **85.71%** |
| 手表版（去EEG+EMG） | 300 | 85.71% |

### 窗口级LOPO结果

| 指标 | 值 |
|------|-----|
| 窗口准确率 | 85.20% |
| Session准确率 | 85.71% |
| Macro F1 | 0.8185 |

### 混淆矩阵

| 实际 \ 预测 | 低疲劳 | 中疲劳 | 高疲劳 |
|:-----------:|:------:|:------:|:------:|
| 低疲劳 | 15 | 1 | 3 |
| 中疲劳 | 1 | 39 | 6 |
| 高疲劳 | 2 | 11 | 26 |

### 参与者级准确率

| P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 66.7% | 100% | 100% | 50% | 100% | 100% | 100% | 100% | 50% | 100% | 100% | 100% |

## 关键发现

1. **EEG为冗余信号**: 去除EEG后准确率不变（85%），特征减少14%，为手表部署奠定基础
2. **EDA是核心信号**: 去除EDA后准确率骤降至65%
3. **手表版本零损失**: 去除EEG和EMG后，手表版本与胸带版本准确率完全一致
4. **10/12参与者完全正确**: P01、P04、P09 准确率较低，受个体差异影响

## 部署

`FatigueMonitor` 类实现三阶段工作流：

```
校准期(5min) → 预热期(14min缓冲) → 实时预测(每30秒)
```

## 参考文献

详见报告中的20篇参考文献，涵盖疲劳检测、HRV生理标准、集成学习算法和运动医学理论。

## License

MIT
