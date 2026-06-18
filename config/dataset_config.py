# -*- coding: utf-8 -*-
"""
WESAD 数据集配置文件
"""
import os
from pathlib import Path

# ── 数据集根目录 ──
# 优先使用环境变量，否则使用项目同级的 WESAD 目录
DATA_ROOT = Path(
    os.environ.get(
        "WESAD_ROOT",
        str(Path(__file__).resolve().parent.parent.parent / "WESAD" / "WESAD")
    )
)

# ── 参与者与会话 ──
NUM_PARTICIPANTS = 12
NUM_SESSIONS_PER_PARTICIPANT = 3
PARTICIPANT_IDS = [f"{i:02d}" for i in range(1, NUM_PARTICIPANTS + 1)]
SESSION_IDS = ["01", "02", "03"]

# ── 活动强度等级（来自 metadata.csv） ──
ACTIVITY_LEVELS = {"low": 0, "medium": 1, "high": 2}
LABEL_NAMES = ["低疲劳", "中疲劳", "高疲劳"]

# ── 核心传感器通道（用于疲劳检测模型） ──
# 选择对疲劳最敏感的信号，避免冗余
CORE_SIGNALS = {
    # 心率相关（来自胸带，1Hz）
    "chest_physiology_summary": {
        "file": "chest_physiology_summary.csv",
        "columns": ["hr", "br", "posture", "hrv"],
        "sampling_rate": 1,
        "device": "Zephyr BioHarness 3.0 (胸带)",
        "description": "心率、呼吸率、姿态、心率变异性",
    },
    # ECG 原始波形（250Hz）
    "chest_raw_ecg": {
        "file": "chest_raw_ecg.csv",
        "columns": ["ecg_waveform"],
        "sampling_rate": 250,
        "device": "Zephyr BioHarness 3.0 (胸带)",
        "description": "原始ECG波形",
    },
    # 手腕心率（1Hz，来自Empatica E4）
    "wrist_hr": {
        "file": "wrist_hr.csv",
        "columns": ["hr"],
        "sampling_rate": 1,
        "device": "Empatica E4 (手腕)",
        "description": "手腕光电心率",
    },
    # BVP 血容量脉搏（64Hz）
    "wrist_bvp": {
        "file": "wrist_bvp.csv",
        "columns": ["bvp"],
        "sampling_rate": 64,
        "device": "Empatica E4 (手腕)",
        "description": "血容量脉搏，可提取心率和HRV",
    },
    # EDA 皮电活动（4Hz）
    "wrist_eda": {
        "file": "wrist_eda.csv",
        "columns": ["eda"],
        "sampling_rate": 4,
        "device": "Empatica E4 (手腕)",
        "description": "皮电活动，反映自主神经兴奋",
    },
    # 皮肤温度（4Hz）
    "wrist_skin_temperature": {
        "file": "wrist_skin_temperature.csv",
        "columns": ["temp"],
        "sampling_rate": 4,
        "device": "Empatica E4 (手腕)",
        "description": "皮肤温度",
    },
    # 手腕加速度（32Hz）
    "wrist_acc": {
        "file": "wrist_acc.csv",
        "columns": ["ax", "ay", "az"],
        "sampling_rate": 32,
        "device": "Empatica E4 (手腕)",
        "description": "手腕三轴加速度",
    },
    # EEG 各频段绝对功率（10Hz，4通道）
    "forehead_eeg_alpha_abs": {
        "file": "forehead_eeg_alpha_abs.csv",
        "columns": ["TP9", "AF7", "AF8", "TP10"],
        "sampling_rate": 10,
        "device": "Muse S (头带)",
        "description": "EEG α频段绝对功率",
    },
    "forehead_eeg_beta_abs": {
        "file": "forehead_eeg_beta_abs.csv",
        "columns": ["TP9", "AF7", "AF8", "TP10"],
        "sampling_rate": 10,
        "device": "Muse S (头带)",
        "description": "EEG β频段绝对功率",
    },
    "forehead_eeg_theta_abs": {
        "file": "forehead_eeg_theta_abs.csv",
        "columns": ["TP9", "AF7", "AF8", "TP10"],
        "sampling_rate": 10,
        "device": "Muse S (头带)",
        "description": "EEG θ频段绝对功率",
    },
    # 呼吸波形（25Hz）
    "chest_raw_breathing": {
        "file": "chest_raw_breathing.csv",
        "columns": ["breathing_waveform"],
        "sampling_rate": 25,
        "device": "Zephyr BioHarness 3.0 (胸带)",
        "description": "原始呼吸波形",
    },
}

# ── 实验标记事件 ──
MARKERS_FILE = "exp_markers.csv"
MARKER_EVENTS = [
    "start_experiment",
    "start_baseline",
    "end_baseline",
    "submit_survey",
    "start_crt",
    "start_nback",
    "start_activity",
    "end_activity",
    "start_fatigue",
    "end_fatigue",
    "end_session",
]

# ── 疲劳标签文件 ──
FATIGUE_FILE = "exp_fatigue.csv"
FATIGUE_COLUMNS = [
    "measurementNumber",
    "physicalFatigueScore",
    "mentalFatigueScore",
    "physicalFatigueAnswerTime",
    "mentalFatigueAnswerTime",
    "fatigueSurveySubmissionTime",
]

# ── 认知任务文件 ──
CRT_FILE = "exp_crt.csv"         # 选择反应时任务
NBACK_FILE = "exp_nback.csv"     # N-back任务

# ── 元数据文件 ──
METADATA_FILE = "metadata.csv"
PRE_TASK_SURVEY_FILE = "pre_task_survey.xlsx"
PRELIMINARY_QUESTIONNAIRE_FILE = "preliminary_questionnaire.xlsx"

# ── 预处理参数 ──
PREPROCESSING = {
    "target_sampling_rate": 32,      # Hz，统一重采样目标频率
    "window_size_seconds": 10,       # 滑动窗口大小（秒）
    "stride_seconds": 5,             # 滑动窗口步长（秒）
    "lowpass_cutoff": 40,            # Hz，低通滤波截止频率
    "highpass_cutoff": 0.5,          # Hz，高通滤波截止频率
}

# ── 模型参数 ──
MODEL_CONFIG = {
    "num_classes": 3,
    "class_names": ["低疲劳", "中疲劳", "高疲劳"],
    "batch_size": 64,
    "learning_rate": 3e-4,
    "epochs": 120,
    "early_stopping_patience": 15,
    "dropout": 0.5,
    "weight_decay": 1e-3,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.3,
}

# ── RAG 配置 ──
RAG_CONFIG = {
    "embedding_model": "text-embedding-3-small",
    "llm_model": "claude-sonnet-4-20250514",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k": 5,
}
