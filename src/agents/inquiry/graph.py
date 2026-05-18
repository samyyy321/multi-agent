from __future__ import annotations
import json
from typing import Any

from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.core.config import get_settings
from src.agents.inquiry.state import (
    InquiryState, InquiryPhase, InquiryHandoffPayload, PatientContext
)

from src.agents.inquiry.confidence import apply_context_weights, check_convergence
from src.agents.inquiry.prompts import (
    CLARIFY_PROMPT, ASK_SYMPTOMS_PROMPT, PARSE_ANSWER_PROMPT,
    CONCLUSION_PROMPT, EMERGENCY_CHECK_PROMPT
)



# ── 依赖注入容器（在 graph 编译时注入，避免全局单例） ──────────────────────
class InquiryDeps:
    def __init__(self, llm, neo4j_driver, embedding_model, milvus_client, db_session):
        self.llm = llm
        self.neo4j_driver = neo4j_driver
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.db_session = db_session


# ════════════════════════════════════════════════════════════════════════
# 节点函数（每个节点接收 state，返回 state 的部分更新）
# ════════════════════════════════════════════════════════════════════════

async def node_load_context(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点①：加载患者上下文。
    从 PostgreSQL 加载既往病史，从 Milvus 加载长期记忆。
    仅在第一轮（round==0）执行，后续轮次跳过。
    """
    if state.round > 0:
        logger.debug("节点①加载患者上下文 非首轮，跳过上下文加载 (round={})", state.round)
        return {}  # 非首轮，不重复加载

    logger.info("节点①加载患者上下文 开始加载患者上下文 | patient_id={} session_id={}",
                state.patient_context.patient_id, state.session_id)

    from src.agents.inquiry.db_queries import load_patient_context
    # user_id 即 patients.id，前端传来的是字符串，转 int 后查患者档案
    patient_id = int(state.patient_context.patient_id) if state.patient_context.patient_id else None
    # 从HIS查询患者信息和就诊记录
    patient_ctx = await load_patient_context(
        patient_id=patient_id,
        db=deps.db_session,
    )

    merged_ctx = PatientContext(
        patient_id=state.patient_context.patient_id,  # 保持 str，贯穿整个流程
        age=patient_ctx.age,
        gender=patient_ctx.gender,
        allergy_history=patient_ctx.allergy_history,
        medical_history=patient_ctx.medical_history,
        long_term_memories=state.patient_context.long_term_memories,
    )
    logger.info("节点①加载患者上下文 上下文加载完成 | age={} gender={} medical_history={} allergy={}",
                merged_ctx.age, merged_ctx.gender,
                len(merged_ctx.medical_history), len(merged_ctx.allergy_history))
    return {"patient_context": merged_ctx}

async def node_check_emergency(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点②：急症识别。
    仅在第一轮执行。识别到急症时直接跳转到 CONCLUDE 阶段，
    并在 candidate_diseases 中放入一个特殊的"急诊"标记。
    """
    if state.round > 0:
        logger.debug("节点②急症识别 非首轮，跳过急症检查")
        return {}

    logger.info("节点②急症识别 开始急症识别检查")
    last_user_msg = ""
    for msg in reversed(state.messages):  # 历史消息： 1-2-3-4-5-6
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {}

    prompt = EMERGENCY_CHECK_PROMPT.format(user_input=last_user_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        if result.get("is_emergency"):
            logger.warning("节点②急症识别 ⚠️ 检测到急症！原因: {}", result.get('reason', ''))
            emergency_reply = (
                "⚠️ 根据您描述的症状，这可能是紧急情况！\n\n"
                f"原因：{result.get('reason', '存在急症风险')}\n\n"
                "**请立即前往最近医院的急诊科就诊，或拨打 120 急救电话。**\n\n"
                "不要等待，请立即行动！"
            )
            return {
                "phase": InquiryPhase.END,
                "messages": [AIMessage(content=emergency_reply)],
            }
        else:
            logger.info("节点②急症识别 未检测到急症，继续正常问诊流程")
    except Exception as e:
        logger.warning(f"急症识别解析失败: {e}")
    return {}
