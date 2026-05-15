from __future__ import annotations
import json
from typing import Any
from src.core.config import get_settings
from loguru import logger
from src.agents.inquiry.state import (
    InquiryState, InquiryPhase, InquiryHandoffPayload, PatientContext
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
