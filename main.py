# -*- coding: utf-8 -*-
"""
FatigueAI 主程序入口
基于 WESAD 数据集的多模态疲劳检测与损伤预防 AI 系统

用法：
    python main.py                # 运行完整流程
    python main.py --check        # 仅检查数据集
    python main.py --train        # 仅训练模型
    python main.py --eval         # 仅评估模型
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

from config.dataset_config import (
    PARTICIPANT_IDS, SESSION_IDS, PREPROCESSING,
    MODEL_CONFIG, CORE_SIGNALS,
)
from data.loader import WESADLoader
from features.extractor import (
    ManualFeatureExtractor, RawSignalWindower,
    DEFAULT_CHANNEL_SPECS,
)
from rag.knowledge_builder import KnowledgeBase
from api.advisor import FatigueAdvisor


def step1_check_dataset():
    """步骤1：数据集快速检查"""
    print("=" * 60)
    print("  步骤1：数据集快速检查")
    print("=" * 60)

    loader = WESADLoader()

    # 元数据
    print("\n[元数据] 活动强度分配:")
    print(loader.metadata.to_string(index=False))

    # 汇总表
    print("\n[会话汇总]:")
    summary = loader.load_all_sessions_summary()
    print(summary.to_string(index=False))
    print(f"\n总计: {len(summary)} 个会话")
    print(f"活动强度分布: {summary['activity_level'].value_counts().to_dict()}")

    # 核心信号采样
    print("\n[核心信号采样] (P01, S01):")
    for name, cfg in CORE_SIGNALS.items():
        try:
            df = loader.load_sensor_data("01", "01", name)
            print(f"  {name}: {len(df):>8} 行 | {cfg['sampling_rate']:>3}Hz | {cfg['device']}")
        except FileNotFoundError:
            print(f"  {name}: 文件缺失")

    # 疲劳标签
    print("\n[疲劳标签] (P01, S01):")
    fatigue = loader.load_fatigue_labels("01", "01")
    print(fatigue.to_string(index=False))

    # 实验标记
    print("\n[实验事件] (P01, S01):")
    markers = loader.load_markers("01", "01")
    print(markers.to_string(index=False))

    return summary


def step2_prepare_features():
    """步骤2：特征工程 — 提取人工特征 + 构建原始信号窗口"""
    print("\n" + "=" * 60)
    print("  步骤2：特征工程")
    print("=" * 60)

    loader = WESADLoader()
    windower = RawSignalWindower()
    extractor = ManualFeatureExtractor()
    fs_map = {n: c["sampling_rate"] for n, c in CORE_SIGNALS.items()}

    # ── 方案A：人工特征（用于 XGBoost） ──
    print("\n[A] 提取人工特征...")
    manual_features_list = []
    manual_labels = []
    manual_groups = []  # 参与者ID（用于Leave-One-Out交叉验证）

    # ── 方案B：原始信号窗口（用于 CNN+LSTM） ──
    print("[B] 构建原始信号窗口...")
    all_windows = []
    all_labels = []
    all_groups = []

    for pid in PARTICIPANT_IDS:
        for sid in SESSION_IDS:
            try:
                label_str = loader.get_activity_label(pid, sid)
                label_int = {"low": 0, "medium": 1, "high": 2}[label_str]

                # 加载传感器数据
                sensor_data = loader.load_all_sensor_data(pid, sid)

                # 方案A：人工特征
                features = extractor.extract_all_features(sensor_data, fs_map)
                manual_features_list.append(features)
                manual_labels.append(label_int)
                manual_groups.append(int(pid))

                # 方案B：原始信号窗口
                multichannel = windower.prepare_multichannel_signal(
                    sensor_data, DEFAULT_CHANNEL_SPECS
                )
                windows, labels = windower.window_and_label(multichannel, label_int)
                all_windows.append(windows)
                all_labels.append(labels)
                all_groups.extend([int(pid)] * len(labels))

                print(f"  P{pid} S{sid} ({label_str}): "
                      f"特征={len(features)}项, 窗口={len(windows)}个")

            except Exception as e:
                print(f"  P{pid} S{sid}: 失败 - {e}")

    # 合并人工特征
    manual_features_df = pd.DataFrame(manual_features_list)
    manual_features_df["label"] = manual_labels
    manual_features_df["participant"] = manual_groups

    # 合并原始信号窗口
    if not all_windows:
        raise RuntimeError("所有会话的数据加载均失败，无法生成任何信号窗口。请检查数据目录是否正确。")
    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_groups = np.array(all_groups)

    print(f"\n人工特征矩阵: {manual_features_df.shape}")
    print(f"原始信号窗口: {all_windows.shape}")
    print(f"标签分布: {np.bincount(all_labels)}")

    return manual_features_df, all_windows, all_labels, all_groups


def step3_train_deep_learning(windows, labels, groups):
    """步骤3：训练深度学习模型（CNN+BiLSTM）"""
    from models.trainer import FatigueTrainer

    print("\n" + "=" * 60)
    print("  步骤3：训练 CNN+BiLSTM 模型")
    print("=" * 60)

    in_channels = windows.shape[2]
    window_size = windows.shape[1]

    print(f"输入通道数: {in_channels}")
    print(f"窗口大小: {window_size} 样本 ({window_size // PREPROCESSING['target_sampling_rate']}秒)")

    # Leave-One-Participant-Out 交叉验证
    logo = LeaveOneGroupOut()
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(windows, labels, groups)
    ):
        test_participant = groups[test_idx][0]
        print(f"\n--- Fold {fold+1}/12: 测试参与者 P{test_participant:02d} ---")

        train_windows, test_windows = windows[train_idx], windows[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]
        train_groups = groups[train_idx]

        trainer = FatigueTrainer(
            model_name="cnn_bilstm",
            in_channels=in_channels,
        )

        result = trainer.train(
            train_windows, train_labels,
            val_groups=train_groups,
            val_ratio=0.2,
        )

        test_metrics = trainer.test(test_windows, test_labels)
        fold_results.append({
            "fold": fold + 1,
            "test_participant": test_participant,
            "val_acc": result["best_val_acc"],
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["f1_macro"],
        })

        print(f"测试准确率: {test_metrics['accuracy']:.4f}, "
              f"F1: {test_metrics['f1_macro']:.4f}")

    # 汇总结果
    results_df = pd.DataFrame(fold_results)
    print("\n" + "=" * 60)
    print("  交叉验证汇总")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\n平均准确率: {results_df['test_acc'].mean():.4f} ± {results_df['test_acc'].std():.4f}")
    print(f"平均F1:     {results_df['test_f1'].mean():.4f} ± {results_df['test_f1'].std():.4f}")

    return results_df


def step4_train_gradient_boosting(features_df):
    """步骤4：训练 Gradient Boosting 基线模型"""
    print("\n" + "=" * 60)
    print("  步骤4：训练 Gradient Boosting 基线模型")
    print("=" * 60)

    feature_cols = [c for c in features_df.columns if c not in ["label", "participant"]]
    X = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["participant"].values  # 直接从DataFrame取，保证长度一致

    # 处理 NaN
    X = np.nan_to_num(X, nan=0.0)

    logo = LeaveOneGroupOut()
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42,
    )

    scores = cross_val_score(model, X, y, groups=groups, cv=logo, scoring="accuracy")
    print(f"LOPO 交叉验证准确率: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"各折结果: {scores}")

    return scores


def step5_demo_advice():
    """步骤5：演示 RAG + 大模型建议生成"""
    print("\n" + "=" * 60)
    print("  步骤5：RAG 知识库 + 建议生成演示")
    print("=" * 60)

    # 初始化知识库和建议生成器
    kb = KnowledgeBase()
    advisor = FatigueAdvisor(knowledge_base=kb, llm_provider="anthropic")

    # 模拟用户画像
    user_profile = {
        "age": 25, "gender": "男", "sport": "篮球",
        "habit": "每周3次，每次90分钟",
        "injury_history": "无",
    }

    # 模拟三种疲劳场景
    scenarios = [
        {
            "name": "低疲劳 - 训练初期",
            "result": {
                "level": "低", "confidence": 0.92, "score": 22,
                "metrics": {"hr": 95, "hrv_rmssd": 45, "eda": 0.2, "temp": 36.5},
            },
        },
        {
            "name": "中疲劳 - 训练中期",
            "result": {
                "level": "中", "confidence": 0.78, "score": 55,
                "metrics": {"hr": 125, "hrv_rmssd": 28, "eda": 0.5, "temp": 36.8},
            },
        },
        {
            "name": "高疲劳 - 训练后期",
            "result": {
                "level": "高", "confidence": 0.88, "score": 82,
                "metrics": {"hr": 148, "hrv_rmssd": 18, "eda": 0.9, "temp": 37.1},
            },
        },
    ]

    for scenario in scenarios:
        print(f"\n{'─' * 50}")
        print(f"场景: {scenario['name']}")
        print(f"{'─' * 50}")
        advice = advisor.generate_advice(
            scenario["result"],
            user_profile=user_profile,
            scene="篮球训练",
            use_llm=False,  # 使用规则引擎（无需API Key）
        )
        print(advice)

    # 保存知识库
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    kb.save_to_json(str(output_dir / "knowledge_base.json"))

    return kb, advisor


def main():
    parser = argparse.ArgumentParser(description="FatigueAI 疲劳检测系统")
    parser.add_argument("--check", action="store_true", help="仅检查数据集")
    parser.add_argument("--train", action="store_true", help="仅训练模型")
    parser.add_argument("--eval", action="store_true", help="仅评估模型")
    args = parser.parse_args()

    start_time = time.time()

    if args.check:
        step1_check_dataset()
        return

    # 完整流程
    step1_check_dataset()
    features_df, windows, labels, groups = step2_prepare_features()

    # 保存特征
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    features_df.to_csv(output_dir / "manual_features.csv", index=False)
    np.save(output_dir / "windows.npy", windows)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "groups.npy", groups)
    print(f"\n特征数据已保存到 {output_dir}/")

    if not args.eval:
        # 训练深度学习模型
        dl_results = step3_train_deep_learning(windows, labels, groups)
        dl_results.to_csv(output_dir / "dl_results.csv", index=False)

        # 训练 Gradient Boosting 基线
        step4_train_gradient_boosting(features_df)

    # RAG + 建议生成演示
    step5_demo_advice()

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
