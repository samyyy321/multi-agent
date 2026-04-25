
from __future__ import annotations

import json
from loguru import logger
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from neo4j import AsyncDriver
from pymilvus import MilvusClient

# ── 第一层：结构化输出模型 ────────────────────────────────────────────────
class SymptomsOutput(BaseModel):
    """LLM 结构化输出的 Schema。with_structured_output 会强制 LLM 按此格式返回。"""
    symptoms: list[str] = Field(
        description="从用户描述中提取并标准化后的医学症状术语列表，无症状时为空列表"
    )


# ── 第一层 Prompt ────────────────────────────────────────────────────────
SYMPTOM_EXTRACT_PROMPT = """你是医疗术语标准化专家。

任务：从用户的描述中提取所有症状，并将每个症状转换为标准医学术语。

标准化规则（不限于此，尽量标准化）：
- 发烧/烧/低烧/高烧 → 发热
- 肚子疼/肚痛/腹部疼痛/肚子不舒服 → 腹痛
- 头晕/头晕眼花/天旋地转 → 眩晕
- 喘不上气/憋气/气短/胸闷喘气 → 呼吸困难
- 拉肚子/跑肚/稀便/大便不成形 → 腹泻
- 心跳快/心慌/心跳加速/心跳不规律 → 心悸
- 浑身没劲/没力气/疲惫/全身乏力 → 乏力
- 嗓子疼/喉咙疼/咽喉痛 → 咽痛
- 胸口疼/胸部疼痛/前胸痛 → 胸痛
- 恶心想吐/想呕吐/胃部不适 → 恶心
- 头疼/头部疼痛/偏头痛 → 头痛
- 流鼻涕/鼻涕/鼻塞流涕 → 流涕
- 咳嗽/干咳/咳痰 → 咳嗽

用户描述：{user_input}

请提取所有症状并标准化后填入 symptoms 字段。如果没有明确症状，symptoms 填空列表。"""


async def extract_and_normalize_symptoms(
    user_input: str,
    llm: BaseChatModel,
) -> list[str]:
    """
    第一层：LLM 提取 + 标准化。
    使用 with_structured_output 强制 LLM 按 SymptomsOutput Schema 返回，
    无需手动解析 JSON，彻底消除格式错误风险。
    """
    structured_llm = llm.with_structured_output(SymptomsOutput)
    prompt = SYMPTOM_EXTRACT_PROMPT.format(user_input=user_input)
    try:
        result: SymptomsOutput = await structured_llm.ainvoke(
            [SystemMessage(content=prompt)]
        )
        symptoms = [s.strip() for s in result.symptoms if s.strip()]
        logger.debug(f"LLM 提取症状: {symptoms}")
        return symptoms
    except Exception as e:
        # with_structured_output 在极少数情况下仍可能失败（如模型不支持 function calling）
        # 此时静默降级，返回空列表，由后续层处理
        logger.warning(f"LLM 结构化输出失败，降级为空列表: {e}")
        return []
