# -*- coding: utf-8 -*-
"""
FatigueAI v4 演示评估
"""
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
import pandas as pd

from config.dataset_config import SESSION_IDS
from data.loader import FatigueSetLoader
from deploy.monitor import FatigueMonitor
from deploy.train import train_and_save, MODELS_DIR


def demo():
    print("=" * 70)
    print("  FatigueAI v4 Demo -- 去EEG + 14min + 子窗口")
    print("=" * 70)

    model_path = MODELS_DIR / "deploy_v4_model.pkl"
    if not model_path.exists():
        print("\n  [训练] 首次运行，训练模型...")
        train_and_save()
    else:
        print(f"\n  [加载] {model_path}")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    print(f"  版本: {bundle.get('version', 'unknown')}")
    print(f"  LOPO窗口准确率: {bundle.get('lopo_window_acc', '?')}")
    print(f"  LOPO Session准确率: {bundle['lopo_accuracy']}, "
          f"F1: {bundle['lopo_f1']}")
    print(f"  EEG去除: {bundle.get('eeg_removed', False)}")
    print(f"  缓冲区: {bundle.get('buffer_sec', '?')}s, "
          f"子窗口: {bundle.get('sub_window_sec', '?')}s")
    if 'pid_accuracies' in bundle:
        print(f"  参与者: {bundle['pid_accuracies']}")

    # P07 Demo
    print("\n  [Demo] P07 测试...")
    loader = FatigueSetLoader()
    demo_pid = "07"

    for sid in SESSION_IDS:
        try:
            label = loader.get_activity_label(demo_pid, sid)
            sensor_data = loader.load_all_sensor_data(demo_pid, sid)
            cleaned = {}
            for name, df in sensor_data.items():
                if len(df) > 0:
                    for col in df.columns:
                        if col != df.columns[0]:
                            df[col] = pd.to_numeric(df[col],
                                                    errors="coerce")
                    cleaned[name] = df

            print(f"\n  --- P{demo_pid} S{sid} (真实: {label}) ---")
            monitor = FatigueMonitor(str(model_path))
            monitor.start_session()

            ref = cleaned.get("chest_physiology_summary")
            if ref is None:
                continue
            ts = ref.iloc[:, 0].values.astype(np.int64) / 1000
            dur = ts[-1] - ts[0]
            t0 = ts[0]

            predictions = []
            first_ready_printed = False
            w_start = 0
            while w_start + 15 <= dur:
                window_data = {}
                for name, df in cleaned.items():
                    ts_col = df.iloc[:, 0].values.astype(np.int64) / 1000
                    mask = ((ts_col >= t0 + w_start)
                            & (ts_col < t0 + w_start + 15))
                    w_df = df.iloc[mask]
                    if len(w_df) > 0:
                        window_data[name] = w_df

                if window_data:
                    result = monitor.update(window_data)
                    predictions.append(result)

                    if not result["ready"]:
                        if (monitor.tick_count in [1, 10, 20, 30]
                                or result.get("phase") == "calibration_complete"):
                            print(f"  [{w_start:>4d}s] "
                                  f"{result.get('message', result['label'])}")
                    else:
                        if (not first_ready_printed
                                or monitor.tick_count % 8 == 0):
                            first_ready_printed = True
                            print(f"  [{w_start:>4d}s] "
                                  f"{result['smoothed_label']} "
                                  f"(conf={result['confidence']:.2f}) "
                                  f"{result['probabilities']}")

                w_start += 15

            label_map = {"low": "低疲劳", "medium": "中疲劳",
                         "high": "高疲劳"}
            label_cn = label_map.get(label, label)
            ready_preds = [p for p in predictions if p["ready"]]
            if ready_preds:
                final = Counter(
                    [p["smoothed_label"]
                     for p in ready_preds]).most_common(1)[0][0]
                correct = "[OK]" if final == label_cn else "[WRONG]"
                print(f"  最终: {final} {correct} "
                      f"(有效预测: {len(ready_preds)}/{len(predictions)})")

        except Exception as e:
            print(f"  [FAIL] P{demo_pid} S{sid}: {e}")

    # 全量评估
    print(f"\n{'=' * 70}")
    print("  全量 LOPO 评估 (v4)")
    print(f"  窗口准确率: {bundle.get('lopo_window_acc', '?')}")
    print(f"  Session准确率: {bundle['lopo_accuracy']}")
    print(f"  F1: {bundle['lopo_f1']}")
    print(f"  混淆矩阵:\n{np.array(bundle['confusion_matrix'])}")
    if 'pid_accuracies' in bundle:
        print(f"  参与者准确率: {bundle['pid_accuracies']}")


if __name__ == "__main__":
    demo()
