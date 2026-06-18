# -*- coding: utf-8 -*-
"""
FatigueAI Web 后端服务
FastAPI + WebSocket 实时数据流
"""
import asyncio
import json
import random
from pathlib import Path
from typing import Dict
import sys
import asyncio

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from data.loader import WESADLoader
from config.dataset_config import PARTICIPANT_IDS, SESSION_IDS, CORE_SIGNALS

app = FastAPI(title="FatigueAI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 数据加载器
loader = WESADLoader()

# ── 缓存加载的数据 ──
_data_cache: Dict[str, Dict] = {}
_cache_lock = asyncio.Lock()


async def load_session_data(pid: str, sid: str) -> Dict:
    """加载并缓存一个会话的数据（异步安全）"""
    key = f"{pid}_{sid}"
    async with _cache_lock:
        if key in _data_cache:
            return _data_cache[key]

    print(f"加载数据: P{pid} S{sid}...")
    signals = {}
    for name, cfg in CORE_SIGNALS.items():
        try:
            df = loader.load_sensor_data(pid, sid, name)
            signals[name] = {
                "data": df,
                "fs": cfg["sampling_rate"],
                "columns": cfg["columns"],
            }
        except Exception as e:
            print(f"  [WARN] 加载信号 {name} 失败: {e}")

    activity_label = loader.get_activity_label(pid, sid)
    fatigue_df = loader.load_fatigue_labels(pid, sid)

    result = {
        "signals": signals,
        "activity_label": activity_label,
        "fatigue_scores": fatigue_df.to_dict("records"),
    }
    async with _cache_lock:
        if key not in _data_cache:
            _data_cache[key] = result
    return _data_cache[key]


# ── 页面路由 ──
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(Path(__file__).parent / "templates" / "index.html"))


# ── API 路由 ──
@app.get("/api/participants")
async def get_participants():
    summary = loader.load_all_sessions_summary()
    return summary.to_dict("records")


@app.get("/api/signal-names")
async def get_signal_names():
    return {k: {"columns": v["columns"], "fs": v["sampling_rate"], "device": v["device"]}
            for k, v in CORE_SIGNALS.items()}


@app.get("/api/fatigue/{pid}/{sid}")
async def get_fatigue_result(pid: str, sid: str):
    """获取疲劳检测结果（模拟模型输出）"""
    data = await load_session_data(pid, sid)
    fatigue_scores = data["fatigue_scores"]

    # 从真实传感器数据提取当前指标
    if fatigue_scores:
        last = fatigue_scores[-1]
        mental_score = last.get("mentalFatigueScore", 50)
        physical_score = last.get("physicalFatigueScore", 50)
        avg_score = (mental_score + physical_score) / 2
    else:
        avg_score = 50
        physical_score = 50
        mental_score = 50

    # 从真实信号数据提取最后时刻的指标值
    def _safe_last(signal_key, col, default):
        if signal_key in data["signals"]:
            df = data["signals"][signal_key]["data"]
            if len(df) > 0 and col in df.columns:
                val = df[col].dropna()
                if len(val) > 0:
                    return round(float(val.values[-1]), 1)
        return default

    hr_val = _safe_last("wrist_hr", "hr", 80)
    hrv_val = _safe_last("chest_physiology_summary", "hrv", 40)
    eda_val = _safe_last("wrist_eda", "eda", 0.3)
    temp_val = _safe_last("wrist_skin_temperature", "temp", 36.5)

    # 映射到等级
    if avg_score < 33:
        level = "低"
    elif avg_score < 66:
        level = "中"
    else:
        level = "高"

    return {
        "level": level,
        "score": round(avg_score, 1),
        "confidence": round(min(0.95, 0.75 + avg_score / 200), 2),
        "physical_score": round(physical_score, 1),
        "mental_score": round(mental_score, 1),
        "activity_label": data["activity_label"],
        "metrics": {
            "hr": hr_val,
            "hrv_rmssd": hrv_val,
            "eda": eda_val,
            "temp": temp_val,
        },
    }


@app.get("/api/feature-importance/{pid}/{sid}")
async def feature_importance(pid: str, sid: str):
    """获取特征重要性（模拟 SHAP 值输出）
    基于当前生理数据计算各特征对疲劳预测的贡献度
    """
    data = await load_session_data(pid, sid)
    signals = data["signals"]
    channels = _extract_channel_data(signals)

    # 基于数据统计特征计算贡献度（模拟 SHAP）
    def calc_contribution(values, normal_low, normal_high):
        if not values:
            return 0.0
        avg = np.mean(values[-100:]) if len(values) > 100 else np.mean(values)
        if avg < normal_low:
            return round(abs(avg - normal_low) / normal_low * 0.8, 3)
        elif avg > normal_high:
            return round(abs(avg - normal_high) / normal_high * 0.9, 3)
        else:
            return round(0.05 + random.uniform(0, 0.1), 3)

    features = []
    feature_map = {
        "心率 (HR)": ("hr", 60, 100),
        "心率变异性 (HRV)": ("hrv", 30, 60),
        "皮电活动 (EDA)": ("eda", 0.05, 0.5),
        "加速度 (ACC)": ("acc", 0.9, 1.1),
        "呼吸频率 (BR)": ("br", 12, 20),
        "皮肤温度": ("temp", 35.5, 37.0),
    }

    for name, (key, lo, hi) in feature_map.items():
        vals = channels.get(key, [])
        contribution = calc_contribution(vals, lo, hi)
        direction = "up" if vals and np.mean(vals[-50:] if len(vals) > 50 else vals) > hi else ("down" if vals and np.mean(vals[-50:] if len(vals) > 50 else vals) < lo else "normal")
        features.append({
            "name": name,
            "key": key,
            "importance": contribution,
            "direction": direction,
            "current_value": round(float(np.mean(vals[-50:] if len(vals) > 50 else vals)), 3) if vals else 0,
        })

    # 按贡献度排序
    features.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "features": features,
        "model": "Stacking Ensemble (4-model)",
        "method": "SHAP (模拟)",
        "session": f"P{pid} S{sid}",
    }


@app.get("/api/model-compare")
async def model_compare():
    """模型性能对比数据（来自真实训练结果）"""
    metrics_path = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "final_v5_results.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 降级：返回默认值
    return {
        "models": [
            {"name": "Stacking Ensemble", "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0},
            {"name": "XGBoost", "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0},
        ],
        "ablation": [],
        "confusion_matrix": [[0,0,0],[0,0,0],[0,0,0]],
        "confusion_labels": ["低", "中", "高"],
        "training_history": {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
    }


# ── WebSocket: 实时传感器数据流（支持循环模式） ──
def _extract_channel_data(signals: Dict) -> Dict[str, list]:
    """从传感器数据中提取各通道"""
    channels = {}
    if "wrist_hr" in signals:
        channels["hr"] = signals["wrist_hr"]["data"]["hr"].values.tolist()
    if "wrist_eda" in signals:
        channels["eda"] = signals["wrist_eda"]["data"]["eda"].values.tolist()
    if "wrist_skin_temperature" in signals:
        channels["temp"] = signals["wrist_skin_temperature"]["data"]["temp"].values.tolist()
    # EEG signals removed in v4 model
    if "chest_physiology_summary" in signals:
        df = signals["chest_physiology_summary"]["data"]
        channels["hrv"] = df["hrv"].values.tolist()
        channels["br"] = df["br"].values.tolist()
    if "wrist_acc" in signals:
        df = signals["wrist_acc"]["data"]
        ax = df["ax"].values
        ay = df["ay"].values
        az = df["az"].values
        channels["acc"] = list(np.sqrt(ax**2 + ay**2 + az**2))
    return channels


@app.websocket("/ws/sensor-stream/{pid}/{sid}")
async def sensor_stream(websocket: WebSocket, pid: str, sid: str):
    """实时数据流，支持循环模式
    query param:
      loop=true    开启循环（默认）
      loop=false   单次播放
      duration=10800  最大持续时间（秒），默认3小时
    """
    await websocket.accept()

    # 读取 query params
    loop_mode = True
    max_duration = 10800  # 3小时
    # FastAPI WebSocket 的 query_params
    qp = dict(websocket.query_params)
    if qp.get("loop", "true").lower() == "false":
        loop_mode = False
    if "duration" in qp:
        try:
            max_duration = int(qp["duration"])
        except ValueError:
            pass

    # 预加载该参与者所有会话数据
    session_data_map: Dict[str, Dict] = {}
    session_order = ["01", "02", "03"]
    for s in session_order:
        try:
            session_data_map[s] = await load_session_data(pid, s)
        except Exception as e:
            print(f"预加载 P{pid} S{s} 失败: {e}")

    if not session_data_map:
        await websocket.send_json({"event": "error", "message": "无可用数据"})
        return

    # 通知前端：循环模式信息
    await websocket.send_json({
        "event": "stream_config",
        "loop": loop_mode,
        "max_duration": max_duration,
        "sessions": list(session_data_map.keys()),
        "start_session": sid,
    })

    global_step = 0
    global_time = 0.0
    cycle_count = 0
    start_ts = asyncio.get_event_loop().time()

    try:
        while True:
            # 检查是否超时
            elapsed = asyncio.get_event_loop().time() - start_ts
            if elapsed >= max_duration:
                await websocket.send_json({"event": "timeout", "message": f"已达到最大时长 {max_duration}s"})
                break

            # 按顺序播放每个会话
            for current_sid in session_order:
                if current_sid not in session_data_map:
                    continue

                data = session_data_map[current_sid]
                signals = data["signals"]
                fatigue_scores = data["fatigue_scores"]
                activity_label = data["activity_label"]
                channels = _extract_channel_data(signals)

                max_len = max(len(v) for v in channels.values()) if channels else 0
                if max_len == 0:
                    continue

                # 通知前端：会话切换
                await websocket.send_json({
                    "event": "session_switch",
                    "session_id": current_sid,
                    "activity_label": activity_label,
                    "cycle": cycle_count,
                    "data_length": max_len,
                })

                for step in range(max_len):
                    # 检查超时
                    elapsed = asyncio.get_event_loop().time() - start_ts
                    if elapsed >= max_duration:
                        await websocket.send_json({"event": "timeout"})
                        return

                    # 疲劳评分模拟
                    progress = step / max(max_len, 1)
                    if fatigue_scores:
                        last_score = fatigue_scores[-1]
                        base_score = (last_score.get("mentalFatigueScore", 50) +
                                      last_score.get("physicalFatigueScore", 50)) / 2
                        fatigue_score = base_score * (0.3 + 0.7 * progress) + random.gauss(0, 3)
                    else:
                        fatigue_score = 30 + 50 * progress + random.gauss(0, 3)
                    fatigue_score = max(0, min(100, fatigue_score))

                    if fatigue_score < 33:
                        fatigue_level = "低"
                    elif fatigue_score < 66:
                        fatigue_level = "中"
                    else:
                        fatigue_level = "高"

                    safe = lambda lst, i, d=0: lst[i] if i < len(lst) else d

                    frame = {
                        "step": global_step,
                        "session_step": step,
                        "session_id": current_sid,
                        "cycle": cycle_count,
                        "time": round(global_time, 1),
                        "session_time": round(step * 1.0, 1),
                        "hr": round(safe(channels.get("hr", []), step, 80) + random.gauss(0, 1), 1),
                        "hrv": round(safe(channels.get("hrv", []), step, 40) + random.gauss(0, 0.5), 1),
                        "eda": round(safe(channels.get("eda", []), step, 0.3) + random.gauss(0, 0.01), 3),
                        "temp": round(safe(channels.get("temp", []), step, 36.5) + random.gauss(0, 0.02), 1),
                        # EEG removed in v4 model
                        "br": round(safe(channels.get("br", []), step, 16) + random.gauss(0, 0.3), 1),
                        "fatigue_score": round(fatigue_score, 1),
                        "fatigue_level": fatigue_level,
                        "confidence": round(random.uniform(0.75, 0.95), 2),
                        "_demo": True,
                        "elapsed": round(elapsed, 1),
                    }

                    await websocket.send_json(frame)
                    global_step += 1
                    global_time += 1.0
                    await asyncio.sleep(0.15)

            cycle_count += 1

            if not loop_mode:
                await websocket.send_json({"event": "end", "cycles": cycle_count})
                break

            # 循环间短暂暂停（1秒），通知前端即将重播
            await websocket.send_json({"event": "cycle_end", "cycle": cycle_count, "message": f"第{cycle_count}轮结束，开始下一轮..."})
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        print(f"客户端断开: P{pid} (共传输 {global_step} 帧, {global_time:.0f}s, {cycle_count} 轮)")
    except Exception as e:
        print(f"WebSocket错误: {e}")
        await websocket.send_json({"event": "error", "message": str(e)})


# ── WebSocket: AI对话 ──
@app.websocket("/ws/chat")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from api.advisor import FatigueAdvisor
    from rag.knowledge_builder import KnowledgeBase

    kb = KnowledgeBase()
    advisor = FatigueAdvisor(knowledge_base=kb, llm_provider="anthropic")

    try:
        while True:
            msg = await websocket.receive_json()
            question = msg.get("question", "")
            fatigue_state = msg.get("fatigue_state", {})

            # 使用规则引擎生成回复（无需API Key）
            context = kb.get_context_for_fatigue(
                fatigue_state.get("level", "中"),
                fatigue_state.get("metrics", {}),
                msg.get("scene", "运动训练"),
            )

            # 简单的问答逻辑
            if "为什么" in question or "原因" in question:
                reply = (
                    "根据您的生理数据分析：\n\n"
                    f"• 心率 {fatigue_state.get('metrics', {}).get('hr', 'N/A')} bpm，"
                    f"高于正常范围，说明交感神经持续激活\n"
                    f"• HRV-RMSSD 下降，表明副交感神经功能减弱\n"
                    f"• 皮电活动升高，反映自主神经兴奋性增加\n\n"
                    "这些变化是身体对持续运动的正常生理反应，但当指标超过阈值时，"
                    "意味着肌肉和神经系统已接近疲劳极限，继续高强度运动会增加损伤风险。"
                )
            elif "恢复" in question or "休息" in question:
                reply = (
                    "基于您当前的疲劳状态，建议的恢复方案：\n\n"
                    "1. **即刻**：停止高强度运动，进行5-10分钟低强度放松（慢走）\n"
                    "2. **30分钟内**：补充碳水化合物（香蕉、运动饮料）和蛋白质\n"
                    "3. **当天**：保证充足水分摄入（每减少1kg体重补1.5L水）\n"
                    "4. **当晚**：保证7-9小时高质量睡眠\n"
                    "5. **次日**：可进行轻度活动（散步、拉伸），避免高强度训练\n"
                    "6. **48小时后**：根据身体感受逐步恢复训练强度"
                )
            elif "预防" in question or "损伤" in question:
                reply = (
                    "疲劳状态下的损伤风险及预防措施：\n\n"
                    "⚠️ **高风险**：\n"
                    "• 膝关节：股四头肌疲劳后膝外翻增大 → ACL损伤风险\n"
                    "• 踝关节：反应时间延长 → 扭伤风险\n"
                    "• 肌肉：代偿发力模式 → 拉伤风险\n\n"
                    "✅ **预防措施**：\n"
                    "• 疲劳评分>60时避免急停变向动作\n"
                    "• 降低运动强度，增加组间休息\n"
                    "• 关注动作规范性，宁可慢不要变形\n"
                    "• 使用护具（护膝、护踝）提供额外保护"
                )
            else:
                reply = (
                    f"您当前疲劳等级：{fatigue_state.get('level', '中')}，"
                    f"评分：{fatigue_state.get('score', 'N/A')}/100\n\n"
                    "您可以问我：\n"
                    "• 「为什么我会疲劳？」— 了解疲劳原因\n"
                    "• 「怎么恢复？」— 获取恢复建议\n"
                    "• 「有什么损伤风险？」— 了解损伤预防\n"
                    "• 「还能继续运动吗？」— 运动建议"
                )

            await websocket.send_json({"reply": reply})

    except WebSocketDisconnect:
        print("聊天客户端断开")
    except Exception as e:
        print(f"聊天错误: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
