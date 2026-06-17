# FatigueAI — 运动疲劳智能监测系统

基于可穿戴生理信号的运动疲劳实时监测系统，通过LOPO交叉验证达到**85.71%**的Session级三分类准确率。

## 核心特性

- **手表可部署**: 仅使用智能手表可采集的6类信号（HR/HRV/EDA/温度/加速度/呼吸率），手表版本零损失迁移
- **多层级特征工程**: 300维特征空间（75基础统计 + 8交互特征 + 225子窗口时序特征）
- **Stacking集成**: XGBoost + LightGBM + CatBoost + RandomForest 四模型集成
- **严格评估**: Leave-One-Participant-Out (LOPO) 交叉验证

## 项目结构

```
FatigueAIProject/
├── config/
│   └── dataset_config.py      # 数据集配置（路径、信号定义、标签映射）
├── data/
│   └── loader.py              # WESAD数据集加载器
├── features/
│   └── extractor.py           # 特征提取器（统计特征、HRV、频域特征）
├── models/
│   ├── trainer.py             # 模型训练（LOPO、Stacking）
│   └── cnn_lstm.py            # CNN+BiLSTM深度学习模型
├── main.py                    # 主入口（数据加载→特征提取→训练→评估）
├── fatigue_deploy_v4.py       # 部署系统（FatigueMonitor实时监测类）
├── requirements.txt           # Python依赖
└── README.md
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行主流程
python main.py

# 运行部署系统
python fatigue_deploy_v4.py
```

## 数据集

使用 [WESAD](https://uni-siegen.sciebo.de/s/HGdUOlNFlFHVkxn) 数据集：
- 12名参与者 × 3次实验 × ~20分钟
- 信号: 胸带（ECG/呼吸）+ 手腕Empatica E4（BVP/EDA/温度/加速度）
- 标签: 三级疲劳状态（低/中/高）

## 实验结果

| 配置 | 特征数 | Session准确率 |
|------|--------|-------------|
| 全特征（含EEG） | 348 | 85.00% |
| 去EEG | 300 | 85.00% |
| 去EDA | 312 | 65.00% |
| **去EEG+stride=30s** | **300** | **85.71%** |
| 手表版（去EEG+EMG） | 300 | 85.71% |

### 参与者级准确率

| 参与者 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 准确率 | 66.7% | 100% | 100% | 50% | 100% | 100% | 100% | 100% | 50% | 100% | 100% | 100% |

## 关键发现

1. **EEG为冗余信号**: 去除EEG后准确率不变（85%），特征减少14%，为手表部署奠定基础
2. **EDA是核心信号**: 去除EDA后准确率骤降至65%
3. **手表版本零损失**: 去除EEG和EMG后，手表版本与胸带版本准确率完全一致

## 部署

`FatigueMonitor` 类实现三阶段工作流：

```
校准期(5min) → 预热期(14min缓冲) → 实时预测(每30秒)
```

## 参考文献

详见报告中的20篇参考文献，涵盖疲劳检测、HRV生理标准、集成学习算法和运动医学理论。

## License

MIT
