# -*- coding: utf-8 -*-
"""
FatigueAI v4 共享特征提取模块 — 去EEG + 子窗口

这是所有 v4 特征提取的单一权威来源。train.py 和 monitor.py
均从此导入，确保训练/部署两端特征一致。
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.dataset_config import CORE_SIGNALS
from features.extractor import ManualFeatureExtractor

# EEG相关特征关键词（v4中去除，filter_eeg_features为安全兜底）
EEG_KEYWORDS = ["eeg_"]


def extract_features(sensor_data):
    """从传感器数据提取特征（v4：去EEG + 非EEG交互特征）

    这是唯一权威的特征提取入口。任何需要提取特征的代码
    （训练、部署、评估）都应调用此函数。
    """
    extractor = ManualFeatureExtractor()
    fs_map = {n: c["sampling_rate"] for n, c in CORE_SIGNALS.items()}
    features = extractor.extract_all_features(sensor_data, fs_map)
    if len(features) < 5:
        return {}

    # Skip EEG features (v4 design: EEG is redundant / cross-participant noise)
    features = {k: v for k, v in features.items()
                if not any(kw in k.lower() for kw in EEG_KEYWORDS)}

    # 交互特征（基于非EEG信号）
    if "hr_mean" in features and "eda_mean" in features:
        if features["eda_mean"] != 0:
            features["hr_eda_ratio"] = features["hr_mean"] / features["eda_mean"]
        features["hr_eda_product"] = features["hr_mean"] * features["eda_mean"]
    if "hr_mean" in features and "temp_mean" in features:
        if features["temp_mean"] != 0:
            features["hr_temp_ratio"] = features["hr_mean"] / features["temp_mean"]
    if "wrist_acc_magnitude_mean" in features and "hr_mean" in features:
        features["acc_hr_interaction"] = \
            features["wrist_acc_magnitude_mean"] * features["hr_mean"]
    if "breathing_mean" in features and "hr_mean" in features:
        if features["breathing_mean"] > 0:
            features["hr_breathing_ratio"] = \
                features["hr_mean"] / features["breathing_mean"]
    if "hrv_sdnn" in features and "hrv_mean_rr" in features:
        if features["hrv_sdnn"] > 0:
            features["hrv_triangular"] = \
                features["hrv_mean_rr"] / features["hrv_sdnn"]
    return features


def filter_eeg_features(feat_dict):
    """去除EEG相关特征（安全兜底，正常情况下extract_features已过滤）"""
    return {k: v for k, v in feat_dict.items()
            if not any(kw in k.lower() for kw in EEG_KEYWORDS)}


def normalize_by_baseline(feat, baseline_feat):
    """相对于基线的变化率归一化：(当前值 - 基线值) / |基线值|"""
    normalized = {}
    for k, v in feat.items():
        bl_v = baseline_feat.get(k, 0.0)
        if bl_v != 0:
            normalized[k] = (v - bl_v) / (abs(bl_v) + 1e-8)
        else:
            normalized[k] = v
    return normalized


def slice_sensor_data(cleaned, t0, start_sec, end_sec):
    """截取时间范围内的传感器数据"""
    window_data = {}
    for name, df in cleaned.items():
        ts_col = df.iloc[:, 0].values.astype(np.int64) / 1000
        mask = (ts_col >= t0 + start_sec) & (ts_col < t0 + end_sec)
        w_df = df.iloc[mask].copy()
        if len(w_df) > 0:
            window_data[name] = w_df
    return window_data


def extract_with_subwindows(sensor_data, baseline_feat, t0, offset,
                            buffer_sec, sub_window_sec):
    """提取主窗口特征 + 子窗口统计特征（v4核心）

    Returns:
        Dict[str, float]: 包含主窗口特征、子窗口 mean/std/delta 的归一化特征字典
    """
    # 主窗口
    window_data = slice_sensor_data(sensor_data, t0, offset,
                                    offset + buffer_sec)
    if not window_data:
        return {}
    full_feat = extract_features(window_data)
    if not full_feat or len(full_feat) < 10:
        return {}

    full_feat = filter_eeg_features(full_feat)
    baseline_feat_filtered = filter_eeg_features(baseline_feat)
    sample = normalize_by_baseline(full_feat, baseline_feat_filtered)

    # 子窗口特征
    if sub_window_sec and sub_window_sec < buffer_sec:
        n_sub = buffer_sec // sub_window_sec
        sub_feats = []
        for i in range(n_sub):
            sw_start = offset + i * sub_window_sec
            sw_data = slice_sensor_data(sensor_data, t0, sw_start,
                                        sw_start + sub_window_sec)
            if sw_data:
                sw_feat = extract_features(sw_data)
                if sw_feat and len(sw_feat) >= 5:
                    sw_feat = filter_eeg_features(sw_feat)
                    sub_feats.append(
                        normalize_by_baseline(sw_feat, baseline_feat_filtered))

        if len(sub_feats) >= 2:
            common_keys = set(sub_feats[0].keys())
            for sf in sub_feats[1:]:
                common_keys &= set(sf.keys())
            for k in common_keys:
                vals = [sf[k] for sf in sub_feats]
                sample[f"sw{sub_window_sec}_{k}_mean"] = np.mean(vals)
                sample[f"sw{sub_window_sec}_{k}_std"] = np.std(vals)
                sample[f"delta{sub_window_sec}_{k}"] = \
                    sub_feats[-1][k] - sub_feats[0][k]

    return sample
