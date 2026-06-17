# src/agents/knowledge/nl2sql.py

from __future__ import annotations
import asyncio
import re
import json
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.agents.knowledge.prompts import NL2SQL_PROMPT, SQL_QA_PROMPT

MAX_SQL_RETRIES = 2
SQL_TIMEOUT_SECONDS = 10

MAX_SQL_ROWS = 100
LIMIT_PATTERN = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)

FORBIDDEN_PATTERNS = [
    re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY|CALL|DO|EXECUTE)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bSELECT\b[\s\S]*\bINTO\b", re.IGNORECASE),
    re.compile(r"\bFOR\s+(UPDATE|NO\s+KEY\s+UPDATE|SHARE|KEY\s+SHARE)\b", re.IGNORECASE),
]
SENSITIVE_PATIENT_FIELD_PATTERN = re.compile(r"\b(phone|id_card)\b", re.IGNORECASE)


def _validate_sql(sql: str) -> tuple[bool, str]:
    """仅允许单条、只读且最多返回 100 行的运营数据查询。"""
    stripped = sql.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1].rstrip()

    if not stripped or ";" in stripped:
        return False, "只允许执行单条 SQL 查询"
    if "--" in stripped or "/*" in stripped or "*/" in stripped:
        return False, "SQL 不允许包含注释"
    if not stripped.upper().startswith("SELECT"):
        return False, "只允许 SELECT 查询"
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(stripped):
            return False, "查询包含禁止的操作"
    if re.search(r"\bPATIENTS\b", stripped, re.IGNORECASE) and SENSITIVE_PATIENT_FIELD_PATTERN.search(stripped):
        return False, "查询包含患者敏感字段"

    limits = [int(value) for value in LIMIT_PATTERN.findall(stripped)]
    if any(limit > MAX_SQL_ROWS for limit in limits):
        return False, f"LIMIT 不得超过 {MAX_SQL_ROWS}"
    if not limits:
        stripped += f" LIMIT {MAX_SQL_ROWS}"
    return True, stripped



async def _generate_sql(
    question: str, llm: BaseChatModel, error_hint: str = "",
) -> str:
    extra = ""
    if error_hint:
        extra = f"\n\n上一次生成的 SQL 执行报错：{error_hint}\n请修正后重新生成。"
    prompt = NL2SQL_PROMPT.format(question=question) + extra
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    sql = response.content.strip()
    if "```" in sql:
        sql = sql.split("```")[1].lstrip("sql").strip()
    return sql

async def _run_sql_query(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
) -> tuple[str | None, list[dict] | str]:
    """生成并执行只读 SQL，返回实际执行的 SQL 与结果或失败提示。"""
    error_hint = ""
    for attempt in range(MAX_SQL_RETRIES + 1):
        raw_sql = await _generate_sql(question, llm, error_hint)
        logger.info(f"NL2SQL SQL (attempt {attempt + 1}): {raw_sql}")

        valid, validated_sql = _validate_sql(raw_sql)
        if not valid:
            logger.warning(f"SQL 安全校验失败: {validated_sql}")
            return None, f"查询被安全策略拦截：{validated_sql}。请换一种方式提问。"

        try:
            query_result = await asyncio.wait_for(
                db.execute(text(validated_sql)),
                timeout=SQL_TIMEOUT_SECONDS,
            )
            rows = query_result.mappings().all()
            return validated_sql, [dict(row) for row in rows[:MAX_SQL_ROWS]]
        except asyncio.TimeoutError:
            logger.warning(f"SQL 执行超时 ({SQL_TIMEOUT_SECONDS}s): {validated_sql}")
            return None, f"查询执行超时（{SQL_TIMEOUT_SECONDS}秒），请简化查询条件后重试。"
        except Exception as error:
            error_hint = str(error)
            logger.warning(f"SQL 执行失败 (attempt {attempt + 1}): {error}")
            if attempt == MAX_SQL_RETRIES:
                return None, "数据查询执行失败，请尝试换一种方式提问。"

    return None, "数据查询执行失败，请尝试换一种方式提问。"


async def search_sql_raw(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
) -> list[dict] | str:
    """执行 NL2SQL 查询并返回原始结构化结果或失败提示。"""
    _, result = await _run_sql_query(question, llm, db)
    return result


async def search_sql(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
) -> str:
    """执行 NL2SQL 查询，并基于实际执行的 SQL 和结果生成回答。"""
    validated_sql, result = await _run_sql_query(question, llm, db)
    if validated_sql is None:
        return result

    if not result:
        return "未查询到相关数据。"

    result_str = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    prompt = SQL_QA_PROMPT.format(
        question=question,
        sql=validated_sql,
        result=result_str,
    )
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content
