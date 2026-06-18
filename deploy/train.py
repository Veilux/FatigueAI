# -*- coding: utf-8 -*-
"""
FatigueAI v4 模型训练模块 — LOPO + 8模型 Stacking
"""
import copy
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
)

from config.dataset_config import PARTICIPANT_IDS, SESSION_IDS, ACTIVITY_LEVELS
from data.loader import WESADLoader
from deploy.features import (
    extract_features, slice_sensor_data, extract_with_subwindows,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# v4 配置常量
BASELINE_SEC = 300       # 校准时长：5分钟
BUFFER_SEC = 840         # 滚动缓冲区：14分钟
STRIDE_SEC = 60          # 预测步长
SUB_WINDOW_SEC = 420     # 子窗口：7分钟


def train_and_save():
    """训练去EEG Stacking模型并保存"""
    print("=" * 70)
    print("  FatigueAI v4 — 去EEG + 14min + 子窗口 Stacking")
    print("=" * 70)

    loader = WESADLoader()
    sessions = []
    for pid in PARTICIPANT_IDS:
        for sid in SESSION_IDS:
            try:
                label_idx = ACTIVITY_LEVELS[loader.get_activity_label(pid, sid)]
                sensor_data = loader.load_all_sensor_data(pid, sid)
                cleaned = {}
                for name, df in sensor_data.items():
                    if len(df) > 0:
                        for col in df.columns:
                            if col != df.columns[0]:
                                df[col] = pd.to_numeric(df[col],
                                                        errors="coerce")
                        cleaned[name] = df
                ref = cleaned.get("chest_physiology_summary")
                if ref is None:
                    continue
                ts = ref.iloc[:, 0].values.astype(np.int64) / 1000
                dur = ts[-1] - ts[0]
                sessions.append({
                    "pid": pid, "sid": sid, "label_idx": label_idx,
                    "cleaned": cleaned, "t0": ts[0], "dur": dur,
                })
                print(f"  [OK] P{pid} S{sid} -> {label_idx} ({dur:.0f}s)")
            except Exception as e:
                print(f"  [跳过] P{pid} S{sid}: {e}")

    # 提取特征（去EEG + 子窗口）
    min_dur = BASELINE_SEC + BUFFER_SEC + STRIDE_SEC
    print(f"\n  提取特征 (基线={BASELINE_SEC}s, 缓冲={BUFFER_SEC}s, "
          f"子窗口={SUB_WINDOW_SEC}s)...")
    all_samples = []
    skipped_dur = []
    skipped_baseline = []
    for s in sessions:
        if s["dur"] < min_dur:
            skipped_dur.append(f"P{s['pid']}S{s['sid']}({s['dur']:.0f}s"
                               f"<{min_dur}s)")
            continue
        baseline_data = slice_sensor_data(s["cleaned"], s["t0"], 0,
                                          BASELINE_SEC)
        baseline_feat = extract_features(baseline_data)
        if not baseline_feat or len(baseline_feat) < 10:
            skipped_baseline.append(f"P{s['pid']}S{s['sid']}")
            continue

        offset = BASELINE_SEC
        while offset + BUFFER_SEC <= s["dur"]:
            sample = extract_with_subwindows(
                s["cleaned"], baseline_feat, s["t0"], offset,
                BUFFER_SEC, SUB_WINDOW_SEC,
            )
            if sample and len(sample) >= 10:
                all_samples.append(
                    (s["pid"], sample, s["label_idx"], s["sid"]))
            offset += STRIDE_SEC

    # 汇报跳过的会话
    if skipped_dur:
        preview = ', '.join(skipped_dur[:5])
        more = ' ...' if len(skipped_dur) > 5 else ''
        print(f"  时长不足跳过 ({len(skipped_dur)}): {preview}{more}")
    if skipped_baseline:
        preview = ', '.join(skipped_baseline[:5])
        more = ' ...' if len(skipped_baseline) > 5 else ''
        print(f"  基线不足跳过 ({len(skipped_baseline)}): {preview}{more}")

    df = pd.DataFrame([s[1] for s in all_samples]).fillna(0).replace(
        [np.inf, -np.inf], 0)
    feature_names = df.columns.tolist()
    X = df.values.astype(np.float64)
    y = np.array([s[2] for s in all_samples])
    groups = np.array([s[0] for s in all_samples])
    sids = np.array([f"{s[0]}_{s[3]}" for s in all_samples])
    print(f"  样本: {len(y)}, 特征: {X.shape[1]} (去EEG后)")

    # 8模型 Stacking（LOPO）
    print("\n  LOPO 训练:")
    unique_pids = np.unique(groups)
    all_probs, all_labels, all_groups, all_sids = [], [], [], []
    all_fold_models = {}

    base_model_defs = _build_base_model_list()
    for test_pid in unique_pids:
        train_mask = groups != test_pid
        test_mask = groups == test_pid
        Xtr_raw, ytr = X[train_mask], y[train_mask]
        Xte_raw, yte = X[test_mask], y[test_mask]

        scaler1 = StandardScaler()
        Xtr_s1 = np.nan_to_num(scaler1.fit_transform(Xtr_raw))
        Xte_s1 = np.nan_to_num(scaler1.transform(Xte_raw))

        scaler2 = QuantileTransformer(output_distribution="normal",
                                      random_state=42)
        Xtr_s2 = np.nan_to_num(scaler2.fit_transform(Xtr_raw))
        Xte_s2 = np.nan_to_num(scaler2.transform(Xte_raw))

        sel = SelectKBest(mutual_info_classif, k=min(50, Xtr_s1.shape[1]))
        Xtr_sel = np.nan_to_num(sel.fit_transform(Xtr_s1, ytr))
        Xte_sel = np.nan_to_num(sel.transform(Xte_s1))

        # 将 feat_ver 映射到实际数据矩阵
        data_map = {"s1": (Xtr_s1, Xte_s1), "s2": (Xtr_s2, Xte_s2),
                    "sel": (Xtr_sel, Xte_sel)}
        base_models = []
        for name, cls, kw, _xtr, _xte, feat_ver in base_model_defs:
            Xtr, Xte = data_map[feat_ver]
            base_models.append((name, cls(**kw), Xtr, Xte, feat_ver))

        # 内层CV Stacking
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        meta_parts = {n: [] for n, _, _, _, _ in base_models}
        meta_y = []
        for tr_idx, val_idx in skf.split(Xtr_s1, ytr):
            meta_y.extend(ytr[val_idx].tolist())
            for name, mt, Xfull, _, _ in base_models:
                m = copy.deepcopy(mt)
                kw_fit = _fit_kwargs(name)
                m.fit(Xfull[tr_idx], ytr[tr_idx], **kw_fit)
                meta_parts[name].append(m.predict_proba(Xfull[val_idx]))

        meta_Xtr = np.hstack([np.vstack(meta_parts[n])
                              for n, _, _, _, _ in base_models])
        meta_ytr = np.array(meta_y)

        # 训练最终基模型 + 生成测试集meta特征
        trained_base = []
        meta_Xte_parts = []
        for name, mt, Xfull, Xte_full, feat_ver in base_models:
            m = copy.deepcopy(mt)
            m.fit(Xfull, ytr, **_fit_kwargs(name))
            meta_Xte_parts.append(m.predict_proba(Xte_full))
            trained_base.append((name, m, feat_ver))
        meta_Xte = np.hstack(meta_Xte_parts)

        meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta_clf.fit(meta_Xtr, meta_ytr)
        probs = meta_clf.predict_proba(meta_Xte)
        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(yte, preds)
        all_probs.append(probs)
        all_labels.extend(yte.tolist())
        all_groups.extend([test_pid] * len(yte))
        all_sids.extend(sids[test_mask].tolist())
        print(f"    P{test_pid}: 窗口={acc:.4f}")

        all_fold_models[test_pid] = {
            "scaler1": scaler1, "scaler2": scaler2, "selector": sel,
            "base_models": trained_base, "meta_clf": meta_clf,
        }

    # 评估指标
    all_probs_arr = np.vstack(all_probs)
    total_acc = accuracy_score(all_labels,
                               np.argmax(all_probs_arr, axis=1))
    total_f1 = f1_score(all_labels, np.argmax(all_probs_arr, axis=1),
                        average="macro", zero_division=0)
    cm = confusion_matrix(all_labels,
                          np.argmax(all_probs_arr, axis=1)).tolist()
    print(f"\n  LOPO 窗口准确率: {total_acc:.4f} (F1={total_f1:.4f})")

    # Session级聚合
    all_sids_arr = np.array(all_sids)
    unique_sids = np.unique(all_sids_arr)
    sess_true, sess_pred = [], []
    for sid in unique_sids:
        mask = all_sids_arr == sid
        sess_true.append(np.array(all_labels)[mask][0])
        sess_pred.append(
            int(np.argmax(np.mean(all_probs_arr[mask], axis=0))))
    sess_acc = accuracy_score(sess_true, sess_pred)

    pid_accs = {}
    for pid in unique_pids:
        ps = {sid for sid in unique_sids if sid.startswith(str(pid) + "_")}
        if ps:
            pm = np.array([s in ps for s in unique_sids])
            if pm.any():
                pid_accs[str(pid)] = round(float(accuracy_score(
                    np.array(sess_true)[pm], np.array(sess_pred)[pm])), 3)
    print(f"  LOPO Session准确率: {sess_acc:.4f}")
    print(f"  混淆矩阵:\n{np.array(cm)}")
    print(f"  参与者: {pid_accs}")

    # 训练通用模型
    print("\n  训练通用模型...")
    scaler1_g = StandardScaler()
    X_s1 = np.nan_to_num(scaler1_g.fit_transform(X))
    scaler2_g = QuantileTransformer(output_distribution="normal",
                                    random_state=42)
    X_s2 = np.nan_to_num(scaler2_g.fit_transform(X))
    sel_g = SelectKBest(mutual_info_classif, k=min(50, X_s1.shape[1]))
    X_sel = np.nan_to_num(sel_g.fit_transform(X_s1, y))

    data_map_g = {"s1": X_s1, "s2": X_s2, "sel": X_sel}
    general_base = []
    for name, cls, kw, _, _, feat_ver in _build_base_model_list():
        m = cls(**kw)
        m.fit(data_map_g[feat_ver], y, **_fit_kwargs(name))
        general_base.append((name, m, feat_ver))

    skf_g = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    meta_parts_g = {n: [] for n, _, _ in general_base}
    meta_y_g = []
    for tr_idx, val_idx in skf_g.split(X_s1, y):
        meta_y_g.extend(y[val_idx].tolist())
        for name, m_template, feat_ver in general_base:
            m = copy.deepcopy(m_template)
            inp = data_map_g[feat_ver]
            m.fit(inp[tr_idx], y[tr_idx], **_fit_kwargs(name))
            meta_parts_g[name].append(m.predict_proba(inp[val_idx]))

    meta_X_g = np.hstack([np.vstack(meta_parts_g[n])
                          for n, _, _ in general_base])
    meta_clf_g = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_clf_g.fit(meta_X_g, np.array(meta_y_g))

    bundle = {
        "version": "v4",
        "feature_names": feature_names,
        "fold_models": all_fold_models,
        "general_model": {
            "scaler1": scaler1_g, "scaler2": scaler2_g,
            "selector": sel_g,
            "base_models": general_base, "meta_clf": meta_clf_g,
        },
        "label_names": ["低疲劳", "中疲劳", "高疲劳"],
        "lopo_window_acc": round(total_acc, 4),
        "lopo_accuracy": round(sess_acc, 4),
        "lopo_f1": round(total_f1, 4),
        "confusion_matrix": cm,
        "pid_accuracies": pid_accs,
        "baseline_sec": BASELINE_SEC,
        "buffer_sec": BUFFER_SEC,
        "sub_window_sec": SUB_WINDOW_SEC,
        "eeg_removed": True,
    }

    path = MODELS_DIR / "deploy_v4_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  模型已保存: {path}")
    return bundle


def _build_base_model_list():
    """构建基模型定义列表"""
    return [
        ("xgb1", XGBClassifier, dict(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
            eval_metric="mlogloss", random_state=42, tree_method="hist"),
         None, None, "s1"),
        ("xgb2", XGBClassifier, dict(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=5,
            gamma=0.3, reg_alpha=1.0, reg_lambda=3.0,
            eval_metric="mlogloss", random_state=123, tree_method="hist"),
         None, None, "s2"),
        ("lgbm1", LGBMClassifier, dict(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
            class_weight="balanced", reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, verbose=-1),
         None, None, "s1"),
        ("lgbm2", LGBMClassifier, dict(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            num_leaves=31, subsample=0.7, colsample_bytree=0.6,
            class_weight="balanced", min_child_samples=10,
            random_state=123, verbose=-1),
         None, None, "sel"),
        ("cb", CatBoostClassifier, dict(
            iterations=200, depth=4, learning_rate=0.1,
            l2_leaf_reg=3.0, auto_class_weights="Balanced",
            random_seed=42, verbose=0),
         None, None, "s1"),
        ("rf", RandomForestClassifier, dict(
            n_estimators=200, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1),
         None, None, "s1"),
        ("gbdt", GradientBoostingClassifier, dict(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_split=10, random_state=42),
         None, None, "sel"),
        ("et", ExtraTreesClassifier, dict(
            n_estimators=200, max_depth=8, min_samples_split=10,
            class_weight="balanced", random_state=42, n_jobs=-1),
         None, None, "s1"),
    ]


def _fit_kwargs(name):
    """返回模型 fit() 的额外关键字参数"""
    known = {"cb": {"verbose": 0}}
    if name in known:
        return known[name]
    if "xgb" in name:
        return {"verbose": False}
    # 已知不需要额外参数
    known_no_kw = {"lgbm1", "lgbm2", "rf", "gbdt", "et"}
    if name not in known_no_kw:
        print(f"  [WARN] 未知模型 {name}，fit() 未传额外参数")
    return {}


if __name__ == "__main__":
    train_and_save()
