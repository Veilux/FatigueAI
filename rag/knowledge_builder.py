# -*- coding: utf-8 -*-
"""
RAG 知识库构建模块
负责将运动疲劳/损伤预防相关知识文档构建为可检索的向量知识库
"""
from pathlib import Path
from typing import List, Dict, Optional
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.dataset_config import RAG_CONFIG


# ── 内置知识库（无需外部文档即可运行） ──
BUILTIN_KNOWLEDGE = [
    # ── 疲劳机制 ──
    {
        "id": "fatigue_mechanism_01",
        "category": "疲劳机制",
        "content": (
            "肌肉疲劳是指肌肉在持续或重复收缩后，产生力量的能力下降的现象。"
            "生理机制包括：(1) ATP消耗和乳酸堆积导致肌肉内环境pH值下降；"
            "(2) 神经肌肉接头处乙酰胆碱释放减少，导致运动单位募集效率降低；"
            "(3) 肌浆网钙离子释放减少，影响肌丝滑行。"
            "肌肉疲劳会导致关节稳定性下降、反应时间延长，显著增加运动损伤风险。"
        ),
    },
    {
        "id": "fatigue_mechanism_02",
        "category": "疲劳机制",
        "content": (
            "精神疲劳（中枢疲劳）是指大脑运动皮层驱动肌肉的能力下降。"
            "表现为：EEG中α波（8-13Hz）和θ波（4-8Hz）功率增加，β波（13-30Hz）功率下降；"
            "心率变异性（HRV）降低；皮电活动（EDA）升高。"
            "精神疲劳会影响注意力、决策能力和运动协调性，间接增加损伤风险。"
        ),
    },
    {
        "id": "fatigue_mechanism_03",
        "category": "疲劳机制",
        "content": (
            "心率变异性（HRV）是评估自主神经系统状态的重要指标。"
            "RMSSD（相邻RR间期差值的均方根）反映副交感神经活性，"
            "疲劳时RMSSD下降表示副交感神经功能减弱、交感神经相对亢进。"
            "LF/HF比值升高也是交感神经激活的标志。"
            "SDNN（RR间期标准差）总体反映心率变异性，疲劳时SDNN通常降低。"
        ),
    },
    # ── 损伤预防 ──
    {
        "id": "injury_prevention_01",
        "category": "损伤预防",
        "content": (
            "运动损伤与疲劳的关联：研究显示约60%的运动损伤发生在训练后半段"
            "（即疲劳积累阶段）。疲劳导致的主要风险因素包括："
            "(1) 肌肉反应时间延长（从正常的120ms增加到200ms以上）；"
            "(2) 关节本体感觉下降，位置觉偏差增大；"
            "(3) 代偿性运动模式出现（如股四头肌疲劳后膝外翻角度增大）；"
            "(4) 核心肌群稳定性下降。"
        ),
    },
    {
        "id": "injury_prevention_02",
        "category": "损伤预防",
        "content": (
            "基于疲劳等级的运动强度调整建议：\n"
            "低疲劳（疲劳评分 0-33）：可继续当前运动强度，注意补充水分。\n"
            "中度疲劳（疲劳评分 34-66）：建议降低运动强度 20-30%，增加组间休息时间，"
            "关注动作规范性，补充碳水化合物和电解质。\n"
            "高度疲劳（疲劳评分 67-100）：强烈建议停止高强度运动，进行5-10分钟低强度"
            "放松活动（如慢走、拉伸），补充蛋白质和碳水化合物，休息至少24小时后再进行"
            "高强度训练。"
        ),
    },
    {
        "id": "injury_prevention_03",
        "category": "损伤预防",
        "content": (
            "运动后恢复策略：\n"
            "1. 主动恢复：运动后进行5-10分钟低强度活动（慢走、慢骑），促进乳酸清除。\n"
            "2. 营养补充：运动后30分钟内摄入碳水化合物（1-1.2g/kg体重）和蛋白质（0.3g/kg）。\n"
            "3. 水分补充：运动前后称重，每减少1kg体重补充1.5L液体。\n"
            "4. 睡眠：保证7-9小时睡眠，睡眠期间生长激素分泌促进肌肉修复。\n"
            "5. 冷热交替浴：促进血液循环，减轻肌肉酸痛。\n"
            "6. 泡沫轴/筋膜放松：缓解肌肉紧张，改善软组织延展性。"
        ),
    },
    # ── 运动科学指南 ──
    {
        "id": "exercise_guideline_01",
        "category": "运动科学",
        "content": (
            "ACSM（美国运动医学会）运动强度建议：\n"
            "低强度：心率储备（HRR）的 20-39%，RPE 9-11，可正常交谈。\n"
            "中强度：HRR 40-59%，RPE 12-13，可说话但不能唱歌。\n"
            "高强度：HRR 60-84%，RPE 14-16，说话困难。\n"
            "极高强度：HRR ≥85%，RPE 17-20，无法持续说话。"
        ),
    },
    {
        "id": "exercise_guideline_02",
        "category": "运动科学",
        "content": (
            "WHO身体活动指南（2020）建议：\n"
            "成人每周至少进行150-300分钟中等强度有氧运动，"
            "或75-150分钟高强度有氧运动，或两者的等量组合。\n"
            "每周至少2天进行肌肉强化活动。\n"
            "限制久坐时间，任何强度的身体活动都比久坐有益。"
        ),
    },
    # ── 特殊人群 ──
    {
        "id": "special_population_01",
        "category": "特殊人群",
        "content": (
            "老年人运动注意事项（65岁以上）：\n"
            "1. 运动前评估：建议进行运动前健康筛查，特别是心血管风险评估。\n"
            "2. 强度调整：建议从中低强度开始，逐步增加。目标心率 = (220-年龄) × 60-70%。\n"
            "3. 平衡训练：每周至少3天进行平衡能力训练，预防跌倒。\n"
            "4. 关节保护：避免高冲击性运动，选择游泳、太极、散步等低冲击活动。\n"
            "5. 水分补充：老年人口渴感减退，需要主动定时补水。"
        ),
    },
    # ── EEG与疲劳 ──
    {
        "id": "eeg_fatigue_01",
        "category": "EEG疲劳指标",
        "content": (
            "EEG频段功率与疲劳的关系：\n"
            "α波（8-13Hz）：闭眼放松时占主导，疲劳时功率增加（尤其在顶叶和枕叶区域）。\n"
            "θ波（4-8Hz）：与困倦和认知负荷增加相关，疲劳时功率显著上升。\n"
            "β波（13-30Hz）：与警觉和注意力相关，疲劳时功率下降。\n"
            "关键指标：α/β比值增加和θ/β比值增加是精神疲劳的可靠神经生理标志物。\n"
            "额叶θ波增加与认知控制下降直接相关。"
        ),
    },
    # ── 生理信号与疲劳 ──
    {
        "id": "physio_fatigue_01",
        "category": "生理信号",
        "content": (
            "皮肤电活动（EDA/皮电）与疲劳/压力的关系：\n"
            "EDA反映交感神经系统活动。精神疲劳和心理压力都会导致EDA升高。\n"
            "皮肤电导水平（SCL）的基线升高表示交感神经持续激活。\n"
            "皮肤电导反应（SCR）的频率和幅度变化可用于评估唤醒水平。\n"
            "疲劳状态下，EDA的变异性通常降低，呈现持续较高的稳态水平。"
        ),
    },
    {
        "id": "physio_fatigue_02",
        "category": "生理信号",
        "content": (
            "皮肤温度与疲劳的关系：\n"
            "运动时核心体温升高，体表血管扩张散热，皮肤温度先升后趋于稳定。\n"
            "持续运动导致体温调节系统疲劳时，皮肤温度可能出现异常波动。\n"
            "体温过高（核心温度>39.5°C）会显著影响认知功能和运动表现。\n"
            "环境温度和湿度会放大疲劳效应，高温高湿环境下应更频繁地监测体温。"
        ),
    },
]


class KnowledgeBase:
    """疲劳运动知识库"""

    def __init__(self, external_docs_dir: Optional[Path] = None):
        self.documents: List[Dict] = []
        self._load_builtin()
        if external_docs_dir and external_docs_dir.exists():
            self._load_external(external_docs_dir)

    def _load_builtin(self):
        """加载内置知识"""
        self.documents.extend(BUILTIN_KNOWLEDGE)
        print(f"已加载 {len(BUILTIN_KNOWLEDGE)} 条内置知识")

    def _load_external(self, docs_dir: Path):
        """加载外部文档（txt/md格式）"""
        count = 0
        for file in docs_dir.glob("*"):
            if file.suffix in [".txt", ".md"]:
                text = file.read_text(encoding="utf-8")
                # 按段落分块
                chunks = self._split_text(text, RAG_CONFIG["chunk_size"])
                for i, chunk in enumerate(chunks):
                    self.documents.append({
                        "id": f"external_{file.stem}_{i}",
                        "category": "外部文档",
                        "content": chunk.strip(),
                    })
                count += 1
        print(f"已加载 {count} 个外部文档，共 {len(self.documents) - len(BUILTIN_KNOWLEDGE)} 条知识块")

    @staticmethod
    def _split_text(text: str, chunk_size: int) -> List[str]:
        """将长文本按段落/句子分块"""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < chunk_size:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current)
                current = para + "\n\n"
        if current:
            chunks.append(current)
        return chunks

    def search(self, query: str, top_k: int = RAG_CONFIG["top_k"]) -> List[Dict]:
        """基于 TF-IDF 风格评分的语义检索（无需外部向量数据库）

        关键词权重 = 词频(TF) × 逆文档频率(IDF) + 类别匹配加成
        """
        import math
        query_lower = query.lower()
        keywords = [kw for kw in query_lower.split() if len(kw) > 1]
        if not keywords:
            keywords = query_lower.split()

        num_docs = len(self.documents)
        scored = []

        for doc in self.documents:
            content_lower = doc["content"].lower()
            score = 0.0

            for kw in keywords:
                # TF: 关键词在文档中出现次数 / 文档总词数
                tf = content_lower.count(kw) / max(1, len(content_lower.split()))
                # IDF: log(总文档数 / 包含该词的文档数)
                doc_count = sum(1 for d in self.documents if kw in d["content"].lower())
                idf = math.log((num_docs + 1) / (doc_count + 1))
                score += tf * idf

            # 类别匹配加成
            for doc_kw in doc["category"].lower().split():
                if doc_kw in query_lower:
                    score += 0.5

            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:top_k]]

    def search_by_category(self, category: str, top_k: int = 5) -> List[Dict]:
        """按类别检索"""
        results = [d for d in self.documents if category in d["category"]]
        return results[:top_k]

    def get_context_for_fatigue(
        self,
        fatigue_level: str,
        metrics: Dict,
        scene: str = "运动训练",
    ) -> str:
        """根据疲劳状态和指标检索相关知识，拼接为上下文字符串"""
        # 构造检索查询
        query_parts = [fatigue_level, scene]

        # 根据关键指标添加查询词（仅当指标值存在且超出正常范围时才检索）
        hr = metrics.get("hr")
        if hr is not None and hr > 120:
            query_parts.append("心率 运动强度")
        hrv_rmssd = metrics.get("hrv_rmssd")
        if hrv_rmssd is not None and hrv_rmssd < 30:
            query_parts.append("心率变异性 HRV 疲劳")
        eda = metrics.get("eda")
        if eda is not None and eda > 0.5:
            query_parts.append("皮电 压力 疲劳")
        eeg_ratio = metrics.get("eeg_alpha_theta_ratio")
        if eeg_ratio is not None and eeg_ratio > 1.5:
            query_parts.append("EEG 脑电 疲劳")

        query = " ".join(query_parts)
        results = self.search(query, top_k=RAG_CONFIG["top_k"])

        context = "以下是从权威运动科学资料中检索到的相关知识：\n\n"
        for i, doc in enumerate(results, 1):
            context += f"[{i}] {doc['category']}: {doc['content']}\n\n"
        return context

    def save_to_json(self, path: str):
        """保存知识库为JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        print(f"知识库已保存: {path} ({len(self.documents)} 条)")

    @classmethod
    def load_from_json(cls, path: str) -> "KnowledgeBase":
        """从JSON加载知识库"""
        kb = cls.__new__(cls)
        with open(path, "r", encoding="utf-8") as f:
            kb.documents = json.load(f)
        return kb


# ── 测试 ──
if __name__ == "__main__":
    kb = KnowledgeBase()
    print(f"\n知识库共 {len(kb.documents)} 条知识")

    # 测试检索
    test_queries = [
        "高强度运动后心率很高怎么办",
        "EEG脑电疲劳信号",
        "老年人运动注意事项",
        "HRV心率变异性降低",
    ]
    for q in test_queries:
        print(f"\n查询: {q}")
        results = kb.search(q, top_k=2)
        for r in results:
            print(f"  [{r['category']}] {r['content'][:80]}...")

    # 测试疲劳上下文生成
    context = kb.get_context_for_fatigue(
        "高",
        {"hr": 145, "hrv_rmssd": 20, "eda": 0.8},
        "篮球训练",
    )
    print(f"\n疲劳建议上下文:\n{context[:500]}...")

    # 保存
    kb.save_to_json("outputs/knowledge_base.json")
