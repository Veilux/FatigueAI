# -*- coding: utf-8 -*-
"""
模型训练与评估流程
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import time
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.dataset_config import MODEL_CONFIG
from models.cnn_lstm import FatigueDataset, get_model


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class FatigueTrainer:
    """疲劳检测模型训练器"""

    def __init__(
        self,
        model_name: str = "cnn_bilstm",
        in_channels: int = 14,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = get_model(model_name, in_channels=in_channels)
        self.model.to(self.device)

        self.config = MODEL_CONFIG
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        print(f"模型: {self.model.__class__.__name__}")
        print(f"设备: {self.device}")
        print(f"参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        val_windows: np.ndarray = None,
        val_labels: np.ndarray = None,
        val_groups: np.ndarray = None,
        val_ratio: float = 0.2,
        augment_train: bool = True,
    ) -> Dict:
        """训练模型

        Args:
            train_windows: 训练窗口 (N, T, C)
            train_labels: 训练标签 (N,)
            val_windows: 预定义的验证窗口（传入则跳过自动切分）
            val_labels: 预定义的验证标签
            val_groups: 参与者ID数组 (N,)，用于按参与者分层切分验证集，避免数据泄露
            val_ratio: 验证集比例（仅自动切分时使用）
            augment_train: 是否启用训练数据增强
        """
        # 划分训练/验证集（按参与者分层，避免同一参与者窗口跨训练/验证集泄露）
        if val_windows is None:
            if val_groups is not None:
                gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
                train_idx, val_idx = next(gss.split(train_windows, train_labels, val_groups))
            else:
                n = len(train_windows)
                val_size = int(n * val_ratio)
                indices = np.random.permutation(n)
                val_idx, train_idx = indices[:val_size], indices[val_size:]
            val_windows, val_labels = train_windows[val_idx], train_labels[val_idx]
            train_windows, train_labels = train_windows[train_idx], train_labels[train_idx]

        train_dataset = FatigueDataset(train_windows, train_labels, augment=augment_train)
        val_dataset = FatigueDataset(val_windows, val_labels, augment=False)

        train_loader = DataLoader(
            train_dataset, batch_size=min(self.config["batch_size"], len(train_dataset)),
            shuffle=True, num_workers=0, drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config["batch_size"],
            shuffle=False, num_workers=0,
        )

        # 类别权重（处理不平衡）
        class_counts = np.bincount(train_labels, minlength=self.config["num_classes"])
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        weight = torch.FloatTensor(class_weights).to(self.device)

        label_smoothing = self.config.get("label_smoothing", 0.0)
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        mixup_alpha = self.config.get("mixup_alpha", 0.0)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        # 学习率预热 + 余弦退火
        warmup_epochs = min(5, self.config["epochs"] // 10)
        scheduler1 = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs,
        )
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["epochs"] - warmup_epochs, eta_min=1e-6,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs],
        )
        early_stopping = EarlyStopping(patience=self.config["early_stopping_patience"])

        best_val_acc = 0
        best_model_state = None

        print(f"\n开始训练 (共 {self.config['epochs']} 轮)")
        print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
        print("-" * 60)

        for epoch in range(self.config["epochs"]):
            # ── 训练阶段 ──
            self.model.train()
            train_loss = 0
            train_preds = []
            train_true = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Mixup 数据增强
                if mixup_alpha > 0 and np.random.random() < 0.5:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    perm = torch.randperm(batch_x.size(0))
                    mixed_x = lam * batch_x + (1 - lam) * batch_x[perm]
                    outputs = self.model(mixed_x)
                    loss = lam * criterion(outputs, batch_y) + (1 - lam) * criterion(outputs, batch_y[perm])
                else:
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_true.extend(batch_y.cpu().numpy())

            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_true, train_preds)

            # ── 验证阶段 ──
            val_loss, val_metrics = self._evaluate(val_loader, criterion)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_metrics["accuracy"])

            # 保存最佳模型
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_model_state = self.model.state_dict().copy()

            scheduler.step()

            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:3d}/{self.config['epochs']} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val F1: {val_metrics['f1_macro']:.4f}"
                )

            # 早停检查
            if early_stopping.step(val_loss):
                print(f"\n早停触发于 Epoch {epoch+1}")
                break

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f}")

        return {
            "best_val_acc": best_val_acc,
            "history": self.history,
        }

    def _evaluate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, Dict]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        }

        return avg_loss, metrics

    def test(self, test_windows: np.ndarray, test_labels: np.ndarray) -> Dict:
        """最终测试评估"""
        test_dataset = FatigueDataset(test_windows, test_labels, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config["batch_size"], shuffle=False)

        criterion = nn.CrossEntropyLoss()
        test_loss, metrics = self._evaluate(test_loader, criterion)

        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())

        metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
        metrics["classification_report"] = classification_report(
            all_labels, all_preds,
            target_names=self.config["class_names"],
            zero_division=0,
        )

        return metrics

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "config": self.config,
            "history": self.history,
        }, path)
        print(f"模型已保存: {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", self.history)
        print(f"模型已加载: {path}")
