# -*- coding: utf-8 -*-
"""
大模型建议生成模块
基于疲劳检测结果 + RAG知识库，调用大模型API生成个性化健康建议
"""
import json
from typing import Dict, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from rag.knowledge_builder import KnowledgeBase

# ── System Prompt ──
SYSTEM_PROMPT = """你是一位专业的运动健康顾问，专门提供疲劳管理和损伤预防建议。

## 核心原则
1. 所有建议必须基于提供的参考资料，不要编造医学建议
2. 如果参考资料中没有相关内容，明确告知用户"建议咨询专业运动医学医生"
3. 建议具体可执行，不要泛泛而谈（如"注意休息"）
4. 根据疲劳等级调整语气：
   - 低疲劳：鼓励性 + 预防性建议
   - 中疲劳：警示性 + 调整建议
   - 高/极高疲劳：严肃警告 + 立即行动建议
5. 涉及严重症状（胸痛、呼吸困难、头晕目眩）时，首要建议是停止运动并就医

## 输出格式
1. 【状态概述】一句话总结当前疲劳状态
2. 【风险分析】结合指标数据说明当前风险（2-3句话）
3. 【行动建议】2-3条，按优先级排列，每条具体可执行
4. 【恢复计划】简短的恢复时间表
5. 【损伤预防】当前状态下需要注意的损伤风险

## 注意
- 回复控制在250字以内，简洁有力
- 使用亲切但专业的语气
- 如用户有伤病历史，在建议中特别提醒"""


class FatigueAdvisor:
    """疲劳健康建议生成器"""

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        api_key: Optional[str] = None,
        llm_provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.kb = knowledge_base or KnowledgeBase()
        self.api_key = api_key
        self.llm_provider = llm_provider
        self.model = model

    def _call_llm(self, system_prompt: str, user_message: str, fatigue_level: str = "中") -> str:
        """调用大模型API"""
        if self.llm_provider == "anthropic":
            return self._call_anthropic(system_prompt, user_message, fatigue_level)
        elif self.llm_provider == "openai":
            return self._call_openai(system_prompt, user_message, fatigue_level)
        else:
            return self._rule_based_fallback(fatigue_level)

    def _call_anthropic(self, system_prompt: str, user_message: str, fatigue_level: str = "中") -> str:
        """调用 Anthropic Claude API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except ImportError:
            print("未安装 anthropic 库，使用规则引擎兜底")
            return self._rule_based_fallback(fatigue_level)
        except Exception as e:
            print(f"API调用失败: {e}，使用规则引擎兜底")
            return self._rule_based_fallback(fatigue_level)

    def _call_openai(self, system_prompt: str, user_message: str, fatigue_level: str = "中") -> str:
        """调用 OpenAI GPT API"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=500,
            )
            return response.choices[0].message.content
        except ImportError:
            print("未安装 openai 库，使用规则引擎兜底")
            return self._rule_based_fallback(fatigue_level)
        except Exception as e:
            print(f"API调用失败: {e}，使用规则引擎兜底")
            return self._rule_based_fallback(fatigue_level)

    def _rule_based_fallback(self, fatigue_level: str) -> str:
        """规则引擎兜底方案（无需API）

        Args:
            fatigue_level: 结构化疲劳等级，取值为 "低"/"中"/"高"/"极高"
        """
        if fatigue_level in ("极高", "高"):
            return (
                "【状态概述】您当前处于高度疲劳状态，身体已接近安全运动阈值。\n"
                "【风险分析】心率和皮电活动升高表明交感神经持续激活，肌肉力量和反应能力下降，"
                "继续高强度运动将显著增加关节和软组织损伤风险。\n"
                "【行动建议】\n"
                "1. 立即停止当前高强度运动\n"
                "2. 进行5-10分钟低强度放松活动（慢走、深呼吸）\n"
                "3. 补充水分和电解质\n"
                "【恢复计划】建议休息至少24小时后再进行高强度训练，今晚保证7-9小时睡眠。\n"
                "【损伤预防】疲劳状态下注意膝关节和踝关节保护，避免急停变向动作。"
            )
        elif fatigue_level == "中":
            return (
                "【状态概述】您当前处于中度疲劳状态，身体机能开始下降。\n"
                "【风险分析】生理指标显示疲劳正在积累，肌肉协调性和反应时间有所下降。\n"
                "【行动建议】\n"
                "1. 适当降低运动强度 20-30%\n"
                "2. 增加组间休息时间，关注动作规范\n"
                "3. 补充碳水化合物和水分\n"
                "【恢复计划】运动后进行5-10分钟拉伸放松，注意当晚充足睡眠。\n"
                "【损伤预防】注意保持正确运动姿势，避免因疲劳导致的代偿性动作。"
            )
        else:
            return (
                "【状态概述】您当前疲劳程度较低，身体状态良好。\n"
                "【风险分析】各项生理指标处于正常范围，可继续当前运动。\n"
                "【行动建议】\n"
                "1. 保持当前运动强度，注意定期补充水分\n"
                "2. 继续关注身体感受，如出现不适及时调整\n"
                "【恢复计划】运动后按常规进行放松和营养补充即可。\n"
                "【损伤预防】运动前做好热身，保持良好的运动习惯。"
            )

    def generate_advice(
        self,
        fatigue_result: Dict,
        user_profile: Optional[Dict] = None,
        scene: str = "运动训练",
        use_llm: bool = True,
    ) -> str:
        """生成个性化疲劳管理建议

        Args:
            fatigue_result: 疲劳检测结果
                {
                    "level": "高",
                    "confidence": 0.85,
                    "score": 78.5,
                    "metrics": {
                        "hr": 145, "hrv_rmssd": 20, "eda": 0.8,
                        "eeg_alpha_theta_ratio": 1.8, "temp": 36.8
                    }
                }
            user_profile: 用户画像（可选）
                {
                    "age": 28, "gender": "男", "sport": "篮球",
                    "habit": "每周3次", "injury_history": "左膝ACL术后"
                }
            scene: 运动场景
            use_llm: 是否使用LLM（False则使用规则引擎）
        """
        level = fatigue_result.get("level", "未知")
        metrics = fatigue_result.get("metrics", {})
        confidence = fatigue_result.get("confidence", 0)

        # RAG 检索相关知识
        knowledge_context = self.kb.get_context_for_fatigue(level, metrics, scene)

        # 构造用户消息
        user_message = f"""
## 用户当前状态
- 疲劳等级：{level}（置信度：{confidence:.0%}）
- 疲劳评分：{fatigue_result.get('score', 'N/A')}/100
- 心率：{metrics.get('hr', 'N/A')} bpm
- HRV-RMSSD：{metrics.get('hrv_rmssd', 'N/A')} ms
- 皮电活动：{metrics.get('eda', 'N/A')} μS
- 皮肤温度：{metrics.get('temp', 'N/A')} °C
- EEG α/θ比值：{metrics.get('eeg_alpha_theta_ratio', 'N/A')}

## 运动场景
{scene}

## 用户画像
{json.dumps(user_profile or {}, ensure_ascii=False)}

## 参考资料（从运动科学知识库检索）
{knowledge_context}

请基于以上信息，给出个性化疲劳管理建议。
"""

        if use_llm:
            return self._call_llm(SYSTEM_PROMPT, user_message, fatigue_level=level)
        else:
            return self._rule_based_fallback(level)

    def batch_generate(
        self,
        results: list,
        scene: str = "运动训练",
    ) -> list:
        """批量生成建议"""
        advices = []
        for result in results:
            advice = self.generate_advice(result, scene=scene)
            advices.append(advice)
        return advices


# ── 测试 ──
if __name__ == "__main__":
    advisor = FatigueAdvisor()

    test_cases = [
        {
            "level": "低",
            "confidence": 0.92,
            "score": 22,
            "metrics": {"hr": 95, "hrv_rmssd": 45, "eda": 0.2, "temp": 36.5},
        },
        {
            "level": "中",
            "confidence": 0.78,
            "score": 55,
            "metrics": {"hr": 125, "hrv_rmssd": 28, "eda": 0.5, "temp": 36.8},
        },
        {
            "level": "高",
            "confidence": 0.88,
            "score": 82,
            "metrics": {"hr": 148, "hrv_rmssd": 18, "eda": 0.9, "temp": 37.1},
        },
    ]

    user = {
        "age": 25, "gender": "男", "sport": "篮球",
        "habit": "每周3次，每次90分钟", "injury_history": "无"
    }

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"疲劳等级: {case['level']} | 评分: {case['score']}")
        print(f"{'='*60}")
        advice = advisor.generate_advice(
            case, user_profile=user, scene="篮球训练", use_llm=False
        )
        print(advice)
