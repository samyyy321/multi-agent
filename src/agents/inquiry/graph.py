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

from src.agents.inquiry.symptom_normalizer import (
    normalize_symptoms
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


async def node_extract_symptoms(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点③：症状标准化。
    从最新的用户消息中提取症状，经三层标准化后合并到 confirmed_symptoms。
    """
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        logger.warning("节点③症状标准化 未找到用户消息，跳过症状提取")
        return {}

    logger.info("节点③症状标准化 开始提取症状 | 用户输入: {!r}", last_user_msg[:80])

    # 等待标准化三层管线执行
    result = await normalize_symptoms(
        user_input=last_user_msg,
        llm=deps.llm,
        neo4j_driver=deps.neo4j_driver,
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
    )

    new_confirmed = list(set(state.confirmed_symptoms) | set(result["all_standard"]))
    new_unmatched = list(set(state.unmatched_symptoms) | set(result["unmatched"]))

    logger.info("节点③症状标准化 提取完成 | 标准化症状={} 未匹配={} 累计确认={}",
                result["all_standard"], result["unmatched"], new_confirmed)

    if new_confirmed:
        return {
            "confirmed_symptoms": new_confirmed,
            "unmatched_symptoms": new_unmatched,
            "phase": InquiryPhase.GRAPH_QUERY, # 接下来进入哪个阶段
        }
    else:
        logger.info("节点③症状标准化 无法提取明确症状，进入澄清流程")
        return {
            "unmatched_symptoms": new_unmatched,
            "phase": InquiryPhase.CLARIFY,
        }

async def node_clarify(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点④：澄清模糊描述。
    当用户描述没有明确症状时，引导用户进一步描述。
    """
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    logger.info("节点④澄清模糊描述 用户描述模糊，发起澄清引导 | round={}", state.round)
    prompt = CLARIFY_PROMPT.format(user_input=last_user_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    logger.debug("节点④澄清模糊描述 澄清回复已生成，等待用户下一轮输入")
    return {
        "round": state.round + 1,
        "messages": [AIMessage(content=response.content)],
    }


async def node_query_neo4j(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点⑤：查询 Neo4j 候选疾病。
    用已确认症状查候选疾病，补充详情，应用上下文权重。
    """
    logger.info("节点⑤查询候选疾病 查询候选疾病 | 确认症状={}", state.confirmed_symptoms)
    candidates = await query_candidate_diseases(
        confirmed_symptoms=state.confirmed_symptoms,
        neo4j_driver=deps.neo4j_driver,
    )
    if candidates:
        candidates = await enrich_candidate_details(candidates, deps.neo4j_driver)
        candidates = apply_context_weights(
            candidates, state.patient_context, state.denied_symptoms
        )
        logger.info("节点⑤查询候选疾病 候选疾病 top5: {}",
                    [(c.name, round(c.confidence, 3)) for c in candidates[:5]])
    else:
        logger.warning("节点⑤查询候选疾病 未找到候选疾病，将强制结束问诊")

    # 收敛判断（要么超10轮，要么有候选疾病，要么没有更多有用信息）
    should_conclude, force_conclude = check_convergence(candidates, state.round)
    logger.info("节点⑤查询候选疾病 收敛判断 | should_conclude={} force_conclude={} round={}",
                should_conclude, force_conclude, state.round)

    if should_conclude:
        return {
            "candidate_diseases": candidates,
            "phase": InquiryPhase.CONCLUDE,
            "force_conclude": force_conclude,
        }
    elif not candidates:
        # 没有候选疾病（症状太罕见），直接结束
        return {
            "candidate_diseases": [],
            "phase": InquiryPhase.CONCLUDE,
            "force_conclude": True,
        }
    else:
        return {
            "candidate_diseases": candidates,
            "phase": InquiryPhase.SYMPTOM_CONFIRM,
        }