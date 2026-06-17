# -*- coding: utf-8 -*-
"""
特征工程模块 — 提供两种特征提取方式：
1. 人工特征提取（用于 XGBoost 基线）
2. 原始信号窗口化（用于 CNN+LSTM 端到端模型）
"""
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.dataset_config import CORE_SIGNALS, PREPROCESSING


class ManualFeatureExtractor:
    """人工特征提取器（用于 XGBoost 等传统 ML 模型）"""

    @staticmethod
    def extract_statistical_features(data: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """提取统计特征：均值、标准差、最大、最小、中位数、偏度、峰度"""
        features = {}
        features[f"{prefix}mean"] = np.nanmean(data)
        features[f"{prefix}std"] = np.nanstd(data)
        features[f"{prefix}max"] = np.nanmax(data)
        features[f"{prefix}min"] = np.nanmin(data)
        features[f"{prefix}median"] = np.nanmedian(data)
        features[f"{prefix}skew"] = skew(data, nan_policy="omit")
        features[f"{prefix}kurtosis"] = kurtosis(data, nan_policy="omit")
        features[f"{prefix}range"] = np.nanmax(data) - np.nanmin(data)
        features[f"{prefix}iqr"] = np.nanpercentile(data, 75) - np.nanpercentile(data, 25)
        return features

    @staticmethod
    def extract_hrv_features(rr_intervals_ms: np.ndarray) -> Dict[str, float]:
        """提取心率变异性（HRV）特征
        Args:
            rr_intervals_ms: R-R间期序列（毫秒）
        """
        features = {}
        # 需要至少5个RR间期才能产生有意义的diff（4个差值和pNN统计）
        # 频域分析需要至少64个RR间期
        if len(rr_intervals_ms) < 5:
            return features
        # 额外保护：丢弃全为零的无效信号
        if np.all(rr_intervals_ms == 0):
            return features

        # 时域特征
        features["hrv_mean_rr"] = np.nanmean(rr_intervals_ms)
        features["hrv_sdnn"] = np.nanstd(rr_intervals_ms)  # SDNN
        mean_rr = np.nanmean(rr_intervals_ms)
        features["hrv_mean_hr"] = 60000.0 / mean_rr if mean_rr > 0 else 0.0  # 平均心率

        # RMSSD（相邻RR间期差值的均方根）
        diff_rr = np.diff(rr_intervals_ms)
        if len(diff_rr) == 0:
            return features
        features["hrv_rmssd"] = np.sqrt(np.nanmean(diff_rr ** 2))

        # pNN50（相邻RR间期差>50ms的百分比）
        nn50 = np.sum(np.abs(diff_rr) > 50)
        features["hrv_pnn50"] = nn50 / len(diff_rr) * 100

        # pNN20
        nn20 = np.sum(np.abs(diff_rr) > 20)
        features["hrv_pnn20"] = nn20 / len(diff_rr) * 100

        # 频域特征（需要足够长的序列）
        if len(rr_intervals_ms) >= 64:
            # 重采样到均匀时间轴（4Hz）
            fs_interp = 4.0
            t_original = np.cumsum(rr_intervals_ms) / 1000.0
            t_uniform = np.arange(0, t_original[-1], 1.0 / fs_interp)
            rr_uniform = np.interp(t_uniform, t_original, rr_intervals_ms)

            # 去趋势
            rr_detrended = rr_uniform - np.convolve(
                rr_uniform, np.ones(int(fs_interp * 30)) / int(fs_interp * 30), mode="same"
            )

            # Welch功率谱
            freqs, psd = scipy_signal.welch(rr_detrended, fs=fs_interp, nperseg=min(256, len(rr_detrended)))

            # VLF (0-0.04Hz), LF (0.04-0.15Hz), HF (0.15-0.4Hz)
            vlf_mask = freqs <= 0.04
            lf_mask = (freqs > 0.04) & (freqs <= 0.15)
            hf_mask = (freqs > 0.15) & (freqs <= 0.4)

            features["hrv_vlf_power"] = float(np.trapz(psd[vlf_mask], freqs[vlf_mask])) if vlf_mask.any() else 0.0
            features["hrv_lf_power"] = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
            features["hrv_hf_power"] = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0

            lf_hf_ratio = features["hrv_lf_power"] / features["hrv_hf_power"] \
                if features["hrv_hf_power"] > 0 else 0.0
            # 防止结果为 Inf
            features["hrv_lf_hf_ratio"] = lf_hf_ratio if np.isfinite(lf_hf_ratio) else 0.0

        return features

    @staticmethod
    def extract_eda_features(eda_signal: np.ndarray, fs: int = 4) -> Dict[str, float]:
        """提取皮电活动（EDA）特征"""
        features = {}
        features["eda_mean"] = np.nanmean(eda_signal)
        features["eda_std"] = np.nanstd(eda_signal)
        features["eda_max"] = np.nanmax(eda_signal)

        # 峰值检测（皮肤电导反应 SCR）
        peaks, _ = scipy_signal.find_peaks(
            eda_signal, height=np.nanmean(eda_signal) + 0.01, distance=int(fs * 1)
        )
        features["eda_num_peaks"] = len(peaks)
        features["eda_peak_rate"] = len(peaks) / (len(eda_signal) / fs)  # 每秒峰值数

        # 强直成分（均值近似）和时相成分（波动幅度）
        features["eda_tonic"] = np.nanmean(eda_signal)
        features["eda_phasic_range"] = np.nanmax(eda_signal) - np.nanmin(eda_signal)

        return features

    @staticmethod
    def extract_eeg_band_features(
        alpha: np.ndarray, beta: np.ndarray, theta: np.ndarray
    ) -> Dict[str, float]:
        """提取EEG频段功率特征（α/β/θ比值与疲劳高度相关）"""
        features = {}

        # 各频段均值（跨通道平均）
        features["eeg_alpha_mean"] = np.nanmean(alpha)
        features["eeg_beta_mean"] = np.nanmean(beta)
        features["eeg_theta_mean"] = np.nanmean(theta)

        # α/θ 比值（疲劳时α和θ增加，比值变化是重要指标）
        theta_safe = np.where(theta == 0, 1e-10, theta)
        features["eeg_alpha_theta_ratio"] = np.nanmean(alpha / theta_safe)

        # α/β 比值（疲劳时α增加、β减少）
        beta_safe = np.where(beta == 0, 1e-10, beta)
        features["eeg_alpha_beta_ratio"] = np.nanmean(alpha / beta_safe)

        # θ/β 比值
        features["eeg_theta_beta_ratio"] = np.nanmean(theta / beta_safe)

        # 各频段功率的标准差（稳定性指标）
        features["eeg_alpha_std"] = np.nanstd(alpha)
        features["eeg_beta_std"] = np.nanstd(beta)
        features["eeg_theta_std"] = np.nanstd(theta)

        return features

    @staticmethod
    def extract_acc_features(acc_data: np.ndarray, prefix: str = "acc_") -> Dict[str, float]:
        """提取加速度计特征（运动强度相关）"""
        features = {}
        # 各轴统计
        for i, axis in enumerate(["x", "y", "z"]):
            if acc_data.ndim > 1 and i < acc_data.shape[1]:
                col = acc_data[:, i]
            else:
                col = acc_data
            features[f"{prefix}{axis}_mean"] = np.nanmean(col)
            features[f"{prefix}{axis}_std"] = np.nanstd(col)

        # 合成加速度幅值
        if acc_data.ndim > 1 and acc_data.shape[1] >= 3:
            magnitude = np.sqrt(acc_data[:, 0]**2 + acc_data[:, 1]**2 + acc_data[:, 2]**2)
            features[f"{prefix}magnitude_mean"] = np.nanmean(magnitude)
            features[f"{prefix}magnitude_std"] = np.nanstd(magnitude)
            features[f"{prefix}sma"] = np.nanmean(magnitude)  # Signal Magnitude Area

        return features

    def extract_all_features(
        self,
        sensor_data: Dict[str, pd.DataFrame],
        fs_map: Dict[str, int],
    ) -> Dict[str, float]:
        """从所有传感器数据中提取全部特征"""
        all_features = {}

        # 心率特征
        try:
            if "chest_physiology_summary" in sensor_data:
                df = sensor_data["chest_physiology_summary"]
                hr = pd.to_numeric(df["hr"], errors="coerce").dropna().values
                hrv = pd.to_numeric(df["hrv"], errors="coerce").dropna().values
                if len(hr) > 10:
                    all_features.update(self.extract_statistical_features(hr, "hr_"))
                if len(hrv) > 10:
                    all_features.update(self.extract_statistical_features(hrv, "chest_hrv_"))
        except Exception as e:
            print(f"  [WARN] 心率特征提取跳过: {type(e).__name__}")

        # HRV from RR intervals
        try:
            if "chest_rr_interval" in sensor_data:
                rr = pd.to_numeric(sensor_data["chest_rr_interval"]["duration"], errors="coerce").dropna().values
                all_features.update(self.extract_hrv_features(rr))
        except Exception as e:
            print(f"  [WARN] HRV特征提取跳过: {type(e).__name__}")

        # BVP-based HR
        try:
            if "wrist_bvp" in sensor_data:
                bvp = pd.to_numeric(sensor_data["wrist_bvp"]["bvp"], errors="coerce").dropna().values
                if len(bvp) > 10:
                    all_features.update(self.extract_statistical_features(bvp, "bvp_"))
        except Exception as e:
            print(f"  [WARN] BVP特征提取跳过: {type(e).__name__}")

        # 手腕心率
        try:
            if "wrist_hr" in sensor_data:
                whr = pd.to_numeric(sensor_data["wrist_hr"]["hr"], errors="coerce").dropna().values
                if len(whr) > 10:
                    all_features.update(self.extract_statistical_features(whr, "wrist_hr_"))
        except Exception as e:
            print(f"  [WARN] 手腕心率特征提取跳过: {type(e).__name__}")

        # EDA
        try:
            if "wrist_eda" in sensor_data:
                eda = pd.to_numeric(sensor_data["wrist_eda"]["eda"], errors="coerce").dropna().values
                if len(eda) > 10:
                    eda_fs = fs_map.get("wrist_eda", 4)
                    all_features.update(self.extract_eda_features(eda, fs=eda_fs))
        except Exception as e:
            print(f"  [WARN] EDA特征提取跳过: {type(e).__name__}")

        # 皮肤温度
        try:
            if "wrist_skin_temperature" in sensor_data:
                temp = pd.to_numeric(sensor_data["wrist_skin_temperature"]["temp"], errors="coerce").dropna().values
                if len(temp) > 10:
                    all_features.update(self.extract_statistical_features(temp, "temp_"))
        except Exception as e:
            print(f"  [WARN] 温度特征提取跳过: {type(e).__name__}")

        # EEG频段
        try:
            eeg_keys = {
                "forehead_eeg_alpha_abs": "alpha",
                "forehead_eeg_beta_abs": "beta",
                "forehead_eeg_theta_abs": "theta",
            }
            eeg_data = {}
            for key, band in eeg_keys.items():
                if key in sensor_data:
                    df = sensor_data[key]
                    vals = []
                    for col in ["TP9", "AF7", "AF8", "TP10"]:
                        if col in df.columns:
                            vals.append(pd.to_numeric(df[col], errors="coerce").dropna().values)
                    if vals:
                        min_len = min(len(v) for v in vals)
                        eeg_data[band] = np.concatenate([v[:min_len] for v in vals])

            if len(eeg_data) == 3:
                min_len = min(len(v) for v in eeg_data.values())
                all_features.update(
                    self.extract_eeg_band_features(
                        eeg_data["alpha"][:min_len],
                        eeg_data["beta"][:min_len],
                        eeg_data["theta"][:min_len],
                    )
                )
        except Exception as e:
            print(f"  [WARN] EEG特征提取跳过: {type(e).__name__}")

        # 加速度
        try:
            if "wrist_acc" in sensor_data:
                df = sensor_data["wrist_acc"]
                cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
                if len(cols) == 3:
                    acc_cols = []
                    for c in cols:
                        acc_cols.append(pd.to_numeric(df[c], errors="coerce").dropna().values)
                    min_len = min(len(v) for v in acc_cols)
                    if min_len > 10:
                        acc = np.stack([v[:min_len] for v in acc_cols], axis=-1)
                        all_features.update(self.extract_acc_features(acc, "wrist_acc_"))
        except Exception as e:
            print(f"  [WARN] 加速度特征提取跳过: {type(e).__name__}")

        # 呼吸
        try:
            if "chest_physiology_summary" in sensor_data:
                br = pd.to_numeric(sensor_data["chest_physiology_summary"]["br"], errors="coerce").dropna().values
                if len(br) > 10:
                    all_features.update(self.extract_statistical_features(br, "breathing_"))
        except Exception as e:
            print(f"  [WARN] 呼吸特征提取跳过: {type(e).__name__}")

        return all_features


class RawSignalWindower:
    """原始信号窗口化（用于 CNN+LSTM 端到端模型）"""

    def __init__(
        self,
        window_seconds: int = PREPROCESSING["window_size_seconds"],
        stride_seconds: int = PREPROCESSING["stride_seconds"],
        target_fs: int = PREPROCESSING["target_sampling_rate"],
    ):
        self.window_size = window_seconds * target_fs  # 样本数
        self.stride = stride_seconds * target_fs
        self.target_fs = target_fs

    def prepare_multichannel_signal(
        self,
        sensor_data: Dict[str, pd.DataFrame],
        channel_specs: List[Tuple[str, str]],  # [(signal_name, column_name), ...]
        session_duration_seconds: Optional[int] = None,
    ) -> np.ndarray:
        """将多个传感器通道对齐并组合成多通道信号
        Returns:
            (num_samples, num_channels) 形状的数组
        """
        channels = []
        for signal_name, col_name in channel_specs:
            if signal_name not in sensor_data:
                continue
            df = sensor_data[signal_name]
            if col_name not in df.columns:
                continue
            data = df[col_name].values

            # 重采样到目标频率
            source_fs = CORE_SIGNALS[signal_name]["sampling_rate"]
            if source_fs != self.target_fs:
                num_target = int(len(data) * self.target_fs / source_fs)
                data = np.interp(
                    np.linspace(0, len(data), num_target),
                    np.arange(len(data)),
                    data,
                )
            channels.append(data)

        if not channels:
            raise ValueError("没有可用的通道数据")

        # 对齐到最短通道
        min_len = min(len(c) for c in channels)
        channels = [c[:min_len] for c in channels]

        return np.stack(channels, axis=-1)  # (num_samples, num_channels)

    def window_and_label(
        self,
        multichannel_signal: np.ndarray,
        label: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """滑动窗口分段，返回 (windows, labels)"""
        num_samples = multichannel_signal.shape[0]
        windows = []
        labels = []

        start = 0
        while start + self.window_size <= num_samples:
            window = multichannel_signal[start:start + self.window_size]
            windows.append(window)
            labels.append(label)
            start += self.stride

        return np.array(windows), np.array(labels)

    @staticmethod
    def augment(window: np.ndarray) -> np.ndarray:
        """数据增强：随机选择一种增强方式"""
        choice = np.random.randint(4)

        if choice == 0:  # 加高斯噪声
            noise = np.random.normal(0, 0.05, window.shape)
            return window + noise

        elif choice == 1:  # 时间缩放
            scale = np.random.uniform(0.9, 1.1)
            n = window.shape[0]
            new_n = int(n * scale)
            scaled = np.interp(
                np.linspace(0, n, new_n), np.arange(n), window[:, 0]
            )
            # 处理多通道
            if window.ndim > 1:
                channels = []
                for c in range(window.shape[1]):
                    ch = np.interp(
                        np.linspace(0, n, new_n), np.arange(n), window[:, c]
                    )
                    channels.append(ch)
                scaled = np.stack(channels, axis=-1)
            # 裁剪或填充回原长度
            if len(scaled) > n:
                return scaled[:n]
            else:
                pad = np.zeros((n - len(scaled),) + scaled.shape[1:])
                return np.concatenate([scaled, pad], axis=0)

        elif choice == 2:  # 通道随机丢弃
            drop_idx = np.random.randint(window.shape[1]) if window.ndim > 1 else 0
            window = window.copy()
            if window.ndim > 1:
                window[:, drop_idx] = 0
            return window

        else:  # 时间偏移
            shift = np.random.randint(-10, 10)
            return np.roll(window, shift, axis=0)


# ── 通道规格定义（用于端到端模型输入） ──
# 选择最具代表性的信号通道
DEFAULT_CHANNEL_SPECS = [
    ("wrist_hr", "hr"),                          # 手腕心率
    ("wrist_eda", "eda"),                         # 皮电活动
    ("wrist_skin_temperature", "temp"),           # 皮肤温度
    ("wrist_acc", "ax"),                          # 加速度X
    ("wrist_acc", "ay"),                          # 加速度Y
    ("wrist_acc", "az"),                          # 加速度Z
    ("forehead_eeg_alpha_abs", "TP9"),            # EEG α (TP9)
    ("forehead_eeg_alpha_abs", "AF7"),            # EEG α (AF7)
    ("forehead_eeg_alpha_abs", "AF8"),            # EEG α (AF8)
    ("forehead_eeg_alpha_abs", "TP10"),           # EEG α (TP10)
    ("forehead_eeg_theta_abs", "TP9"),            # EEG θ (TP9)
    ("forehead_eeg_theta_abs", "AF7"),            # EEG θ (AF7)
    ("forehead_eeg_beta_abs", "TP9"),             # EEG β (TP9)
    ("forehead_eeg_beta_abs", "AF7"),             # EEG β (AF7)
]
