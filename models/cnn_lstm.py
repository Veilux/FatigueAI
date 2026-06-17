# -*- coding: utf-8 -*-
"""
CNN + BiLSTM 疲劳检测模型
自动学习多模态生理信号的时序特征，输出疲劳等级分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.dataset_config import MODEL_CONFIG


class FatigueDataset(Dataset):
    """疲劳检测 PyTorch Dataset"""

    def __init__(
        self,
        windows: np.ndarray,    # (num_windows, window_size, num_channels)
        labels: np.ndarray,     # (num_windows,)
        augment: bool = False,
    ):
        self.windows = torch.FloatTensor(windows).permute(0, 2, 1)  # (N, C, T)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        y = self.labels[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """训练时数据增强"""
        choice = torch.randint(0, 4, (1,)).item()

        if choice == 0:  # 加高斯噪声
            noise = torch.randn_like(x) * 0.05
            x = x + noise

        elif choice == 1:  # 随机缩放
            scale = torch.empty(1).uniform_(0.9, 1.1).item()
            n = x.shape[1]
            new_n = int(n * scale)
            x = F.interpolate(x.unsqueeze(0), size=new_n, mode="linear", align_corners=False).squeeze(0)
            if x.shape[1] > n:
                x = x[:, :n]
            else:
                pad = torch.zeros(x.shape[0], n - x.shape[1])
                x = torch.cat([x, pad], dim=1)

        elif choice == 2:  # 通道随机丢弃
            drop_ch = torch.randint(0, x.shape[0], (1,)).item()
            x = x.clone()
            x[drop_ch] = 0

        else:  # 时间偏移
            shift = torch.randint(-10, 10, (1,)).item()
            x = torch.roll(x, shift, dims=1)

        return x


class CNNBiLSTM(nn.Module):
    """CNN + 双向LSTM 疲劳检测网络（轻量化版本，减少过拟合）

    结构：
    ┌─────────────────────────────────────────────┐
    │ 输入: (batch, num_channels, window_size)     │
    │         ↓                                    │
    │ Conv1d Block 1: (C→32,  k=7) + BN + ReLU    │
    │ Dropout1d(0.1) + MaxPool1d(2)                │
    │         ↓                                    │
    │ Conv1d Block 2: (32→64, k=5) + BN + ReLU    │
    │ Dropout1d(0.15) + MaxPool1d(2)               │
    │         ↓                                    │
    │ Conv1d Block 3: (64→128, k=3) + BN + ReLU   │
    │ Dropout1d(0.2) + MaxPool1d(2)                │
    │         ↓                                    │
    │ BiLSTM × 2层 (128→64×2)                      │
    │         ↓                                    │
    │ Dropout(0.6) → FC(128→64) → FC(64→num_cls)  │
    └─────────────────────────────────────────────┘
    """

    def __init__(
        self,
        in_channels: int = 14,
        num_classes: int = MODEL_CONFIG["num_classes"],
        dropout: float = MODEL_CONFIG["dropout"],
    ):
        super().__init__()

        # ── CNN 特征提取器（轻量化，减少过拟合） ──
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.1),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.15),
            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(kernel_size=2),
        )

        # ── BiLSTM 时序建模 ──
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True,
        )

        # ── 分类头 ──
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_channels, window_size)
        Returns:
            logits: (batch, num_classes)
        """
        # CNN: (B, C, T) → (B, 128, T/8)
        x = self.cnn(x)

        # 维度转换给LSTM: (B, 128, T/8) → (B, T/8, 128)
        x = x.permute(0, 2, 1)

        # BiLSTM: (B, T/8, 128) → (B, T/8, 128)
        x, _ = self.lstm(x)

        # 取最后时刻的隐藏状态
        x = x[:, -1, :]  # (B, 128)

        # 分类
        x = self.classifier(x)  # (B, num_classes)
        return x

    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取CNN特征（用于可视化和分析）"""
        with torch.no_grad():
            return self.cnn(x)


class CNNSimple(nn.Module):
    """纯CNN消融实验模型（去掉LSTM）"""

    def __init__(self, in_channels=14, num_classes=3, dropout=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x).squeeze(-1)
        return self.fc(x)


class LSTMSimple(nn.Module):
    """纯LSTM消融实验模型（去掉CNN）"""

    def __init__(self, in_channels=14, num_classes=3, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels, hidden_size=128,
            num_layers=2, batch_first=True,
            dropout=0.3, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1)
        return x * w


class MultiScaleBlock(nn.Module):
    """多尺度卷积块：并行使用不同核大小捕获不同时间尺度的模式"""
    def __init__(self, in_ch, out_ch_per_branch=24):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch_per_branch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch_per_branch),
                nn.ReLU(inplace=True),
            )
            for k in [3, 5, 7, 11]
        ])
        self.total_ch = out_ch_per_branch * 4
        self.se = SEBlock(self.total_ch, reduction=4)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.se(out)
        return out


class TemporalAttention(nn.Module):
    """时间步注意力：学习关注最重要的时间片段"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, lstm_out):
        # lstm_out: (B, T, H)
        weights = self.attention(lstm_out)         # (B, T, 1)
        weights = torch.softmax(weights, dim=1)    # (B, T, 1)
        context = (lstm_out * weights).sum(dim=1)  # (B, H)
        return context


class CNNBiLSTMv2(nn.Module):
    """多尺度注意力 CNN+BiLSTM — 针对跨参与者泛化优化

    相比 v1 的改进：
    1. 多尺度并行卷积（核大小 3/5/7/11）— 捕获不同时间尺度的生理模式
    2. SE通道注意力 — 自适应学习各信号通道的重要性
    3. 时间步注意力 — 关注 LSTM 输出中最关键的时间片段
    4. 残差连接 — 改善梯度流动
    5. 更强正则化 — 对抗过拟合
    """

    def __init__(
        self,
        in_channels: int = 14,
        num_classes: int = MODEL_CONFIG["num_classes"],
        dropout: float = MODEL_CONFIG["dropout"],
    ):
        super().__init__()

        # Stage 1: 多尺度特征提取
        self.stage1 = nn.Sequential(
            MultiScaleBlock(in_channels, out_ch_per_branch=24),  # -> 96 ch
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(0.1),
        )
        # 残差投影（通道数变化时需要）
        self.residual_proj1 = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=1),
            nn.BatchNorm1d(96),
        )
        self.pool_res1 = nn.MaxPool1d(kernel_size=2)

        # Stage 2: 更深的多尺度特征
        self.stage2 = nn.Sequential(
            MultiScaleBlock(96, out_ch_per_branch=24),  # -> 96 ch
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(0.15),
        )
        self.pool_res2 = nn.MaxPool1d(kernel_size=2)

        # Stage 3: 精细化 + 降维
        self.stage3 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(kernel_size=2),
        )

        # BiLSTM 时序建模
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=80,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        # 时间步注意力
        self.temporal_attn = TemporalAttention(160)  # 80*2 = 160

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(160, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) = (B, 14, 320)

        # Stage 1 with residual
        identity = self.pool_res1(self.residual_proj1(x))
        x = self.stage1(x)
        x = x + identity  # 残差连接

        # Stage 2 with residual (needs pooling to match spatial dims)
        identity = self.pool_res2(x)
        x = self.stage2(x)
        x = x + identity  # 残差连接

        # Stage 3
        x = self.stage3(x)

        # BiLSTM: (B, 128, T') -> (B, T', 128) -> (B, T', 160)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        # 时间步注意力（代替只取最后时刻）
        x = self.temporal_attn(x)  # (B, 160)

        # 分类
        x = self.classifier(x)
        return x


def get_model(model_name: str = "cnn_bilstm", **kwargs) -> nn.Module:
    """模型工厂函数"""
    models = {
        "cnn_bilstm": CNNBiLSTM,
        "cnn_bilstm_v2": CNNBiLSTMv2,
        "cnn_only": CNNSimple,
        "lstm_only": LSTMSimple,
    }
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}, 可选: {list(models.keys())}")
    return models[model_name](**kwargs)


def print_model_summary(model: nn.Module, input_channels: int = 14, window_size: int = 320):
    """打印模型结构摘要"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型: {model.__class__.__name__}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 测试前向传播
    dummy = torch.randn(2, input_channels, window_size)
    with torch.no_grad():
        output = model(dummy)
    print(f"输入形状: {dummy.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出类别概率: {F.softmax(output, dim=1)}")
