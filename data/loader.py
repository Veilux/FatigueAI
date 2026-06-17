# -*- coding: utf-8 -*-
"""
FatigueSet 数据加载与预处理模块
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.dataset_config import (
    DATA_ROOT, PARTICIPANT_IDS, SESSION_IDS,
    CORE_SIGNALS, METADATA_FILE, FATIGUE_FILE,
    MARKERS_FILE, PREPROCESSING
)


class FatigueSetLoader:
    """FatigueSet 数据集加载器"""

    def __init__(self, data_root: Path = DATA_ROOT):
        self.data_root = data_root
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> pd.DataFrame:
        """加载元数据（活动强度分配表）"""
        path = self.data_root / METADATA_FILE
        df = pd.read_csv(path)
        # 统一列名为小写
        df.columns = df.columns.str.strip().str.lower()
        # 将 participant_id 统一为零填充字符串（如 1 / 1.0 → "01"）
        df["participant_id"] = df["participant_id"].apply(lambda x: str(int(float(x))).zfill(2))
        # 将 session 列也统一为零填充字符串
        for col in ["low_session", "medium_session", "high_session"]:
            df[col] = df[col].apply(lambda x: str(int(float(x))).zfill(2))
        return df

    def get_activity_label(self, participant_id: str, session_id: str) -> str:
        """获取指定参与者和会话的活动强度标签"""
        pid = str(participant_id).zfill(2)
        sid = str(session_id).zfill(2)
        row = self.metadata[self.metadata["participant_id"] == pid]
        if row.empty:
            raise ValueError(f"找不到参与者 {participant_id}")
        row = row.iloc[0]
        if sid == row["low_session"]:
            return "low"
        elif sid == row["medium_session"]:
            return "medium"
        elif sid == row["high_session"]:
            return "high"
        else:
            raise ValueError(f"会话 {sid} 与参与者 {pid} 不匹配")

    def load_sensor_data(
        self,
        participant_id: str,
        session_id: str,
        signal_name: str,
    ) -> pd.DataFrame:
        """加载单个传感器信号文件"""
        signal_config = CORE_SIGNALS[signal_name]
        file_path = self.data_root / participant_id / session_id / signal_config["file"]

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df

    def load_fatigue_labels(self, participant_id: str, session_id: str) -> pd.DataFrame:
        """加载疲劳主观评分（多次测量）"""
        path = self.data_root / participant_id / session_id / FATIGUE_FILE
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df

    def load_markers(self, participant_id: str, session_id: str) -> pd.DataFrame:
        """加载实验事件标记"""
        path = self.data_root / participant_id / session_id / MARKERS_FILE
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df

    def load_all_sessions_summary(self) -> pd.DataFrame:
        """生成所有会话的汇总表"""
        records = []
        for pid in PARTICIPANT_IDS:
            for sid in SESSION_IDS:
                try:
                    label = self.get_activity_label(pid, sid)
                    fatigue_df = self.load_fatigue_labels(pid, sid)
                    # 取最后一次疲劳评分作为最终疲劳状态
                    last = fatigue_df.iloc[-1]
                    records.append({
                        "participant_id": pid,
                        "session_id": sid,
                        "activity_level": label,
                        "activity_label": {"low": 0, "medium": 1, "high": 2}[label],
                        "physical_fatigue_score": last["physicalFatigueScore"],
                        "mental_fatigue_score": last["mentalFatigueScore"],
                        "num_measurements": len(fatigue_df),
                    })
                except Exception as e:
                    print(f"警告: P{pid} S{sid} 加载失败 - {e}")
        return pd.DataFrame(records)

    def load_all_sensor_data(
        self,
        participant_id: str,
        session_id: str,
        signal_names: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """加载一个会话的所有（或指定）传感器数据"""
        if signal_names is None:
            signal_names = list(CORE_SIGNALS.keys())

        data = {}
        for name in signal_names:
            try:
                data[name] = self.load_sensor_data(participant_id, session_id, name)
            except FileNotFoundError:
                print(f"警告: {participant_id}/{session_id} 缺少信号 {name}")
        return data


class SignalPreprocessor:
    """信号预处理工具"""

    def __init__(self, target_fs: int = PREPROCESSING["target_sampling_rate"]):
        self.target_fs = target_fs

    def resample(self, df: pd.DataFrame, source_fs: int, target_fs: int = None) -> pd.DataFrame:
        """将信号重采样到目标频率"""
        if target_fs is None:
            target_fs = self.target_fs

        if source_fs == target_fs:
            return df

        # 基于时间戳的线性插值重采样
        ts = df.iloc[:, 0].values  # 时间戳列
        data_cols = df.columns[1:]

        # 计算目标时间戳
        duration_ms = ts[-1] - ts[0]
        num_target_samples = int(duration_ms * target_fs / 1000)
        target_ts = np.linspace(ts[0], ts[-1], num_target_samples)

        # 对每个数据列进行插值
        resampled = {"timestamp": target_ts}
        for col in data_cols:
            resampled[col] = np.interp(target_ts, ts, df[col].values)

        return pd.DataFrame(resampled)

    def normalize_per_subject(self, data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Z-score 标准化（按被试个体基线）"""
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std == 0:
            return np.zeros_like(data), mean, std
        return (data - mean) / std, mean, std

    def sliding_window(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int,
    ) -> np.ndarray:
        """滑动窗口分段
        Args:
            data: (num_samples, num_channels) 形状的信号
            window_size: 窗口大小（样本数）
            stride: 步长（样本数）
        Returns:
            (num_windows, window_size, num_channels) 形状的数组
        """
        num_samples = data.shape[0]
        windows = []
        start = 0
        while start + window_size <= num_samples:
            windows.append(data[start:start + window_size])
            start += stride
        return np.array(windows)

    def segment_by_markers(
        self,
        signal_df: pd.DataFrame,
        markers_df: pd.DataFrame,
        start_event: str,
        end_event: str,
    ) -> pd.DataFrame:
        """根据实验标记截取指定阶段的信号"""
        start_ts = markers_df[markers_df["eventMarker"] == start_event]["utcTime"].values
        end_ts = markers_df[markers_df["eventMarker"] == end_event]["utcTime"].values

        if len(start_ts) == 0 or len(end_ts) == 0:
            raise ValueError(f"找不到标记 {start_event} 或 {end_event}")

        ts = signal_df.iloc[:, 0].values
        mask = (ts >= start_ts[0]) & (ts <= end_ts[0])
        return signal_df[mask].reset_index(drop=True)


# ── 快速验证数据加载 ──
def quick_check():
    """快速验证数据集加载是否正常"""
    print("=" * 60)
    print("FatigueSet 数据集快速检查")
    print("=" * 60)

    loader = FatigueSetLoader()

    # 1. 元数据
    print("\n[1] 元数据（活动强度分配）:")
    print(loader.metadata.to_string(index=False))

    # 2. 汇总表
    print("\n[2] 所有会话汇总:")
    summary = loader.load_all_sessions_summary()
    print(summary.to_string(index=False))
    print(f"\n总计: {len(summary)} 个会话")
    print(f"活动强度分布: {summary['activity_level'].value_counts().to_dict()}")

    # 3. 传感器数据采样
    print("\n[3] 传感器数据采样 (P01, S01):")
    for name in ["wrist_hr", "wrist_eda", "forehead_eeg_alpha_abs", "chest_physiology_summary"]:
        if name in CORE_SIGNALS:
            try:
                df = loader.load_sensor_data("01", "01", name)
                cfg = CORE_SIGNALS[name]
                print(f"  {name}: {len(df)} 行, {cfg['sampling_rate']}Hz, 列: {list(df.columns)}")
            except Exception as e:
                print(f"  {name}: 加载失败 - {e}")

    # 4. 疲劳标签
    print("\n[4] 疲劳标签 (P01, S01):")
    fatigue = loader.load_fatigue_labels("01", "01")
    print(fatigue.to_string(index=False))

    # 5. 实验标记
    print("\n[5] 实验事件标记 (P01, S01):")
    markers = loader.load_markers("01", "01")
    print(markers.to_string(index=False))


if __name__ == "__main__":
    quick_check()
