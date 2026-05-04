from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.redis import AsyncRedisSaver
from langchain.agents import create_agent
from langchain_deepseek.chat_models import ChatDeepSeek

from src.agents.store_tools import save_memory, search_memory
from src.core.config import get_settings
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.milvus_store import MilvusStore
from src.infra.redis_cache import get_checkpointer_redis
from dotenv import load_dotenv
from src.agents.worker_tools import WORKER_TOOLS

load_dotenv()
settings = get_settings()

SUPERVISOR_SYSTEM_PROMPT = """你是医疗系统的智能总助手（Supervisor）。

你的核心职责：
1. 与患者/医生/运营人员进行多轮对话
2. 准确识别用户意图，将任务分派给合适的专项助手
3. 整合专项助手的结果，给出清晰、友好的最终回复
4. 主动收集必要信息（如症状描述不清时追问）
5. 管理对话上下文，保持对话连贯性

可调用的专项助手：
- call_inquiry_agent：智慧问诊（症状分诊、挂号建议）
- call_knowledge_agent：医学知识问答（疾病科普、治疗方案）

记忆工具：
- save_memory：将重要信息（病史、过敏史、用药偏好等）保存到长期记忆
- search_memory：从长期记忆中检索用户历史信息

工作原则：
- 优先从长期记忆中检索用户历史信息，避免重复询问
- 遇到复杂问题可以串联多个专项助手（先问诊再查药）
- 始终以患者安全为第一优先级
- 对话语气温和、专业、易懂"""

# 剩余忽略：主要是提示词和工具
 # ── 工具 = 记忆工具 + Worker 工具 ─────────────────────────────────
tools = [save_memory, search_memory] + WORKER_TOOLS


def _get_embedding_model():
    """返回向量化模型。根据你的实际情况替换。"""
    # 方案A：使用 DashScope（阿里云）
    from langchain_community.embeddings import DashScopeEmbeddings
    return DashScopeEmbeddings(model=settings.EMBEDDING_MODEL, 
            dashscope_api_key=settings.DASHSCOPE_API_KEY
        )

async def create_supervisor_agent():
    """
    创建带有短期记忆（Redis）和长期记忆（Milvus）的 Supervisor Agent。

    短期记忆：Redis checkpointer，保存当前会话的完整对话历史
    长期记忆：Milvus store，跨会话的语义记忆，Agent 通过工具主动读写
    """

    # ── 短期记忆：Redis Checkpointer（复用 infra 层连接）─────────────

    # 1. 复用项目已有的 checkpointer 专用 Redis 客户端（bytes 模式）
    redis_client = get_checkpointer_redis()

    # 2. 创建 AsyncRedisSaver，并调用 asetup() 初始化 RediSearch 索引
    # asetup() 会在 Redis Stack 中创建 checkpoint / checkpoint_write 两个索引
    # 必须在首次使用前调用一次，索引已存在时自动跳过，可以重复调用
    checkpointer = AsyncRedisSaver(redis_client=redis_client)
    await checkpointer.asetup()


    # ── 长期记忆：Milvus Store ─────────────────────────────────────────
    milvus_alias = get_milvus_client_alias()
    embedding_model = _get_embedding_model()
    store = MilvusStore(
        alias=milvus_alias,
        embeddings=embedding_model,
        dims=1024,   # DashScope text-embedding-v3 默认输出 1024 维
    )

    # ── 工具列表 ───────────────────────────────────────────────────────
    tools = [
        save_memory,  # 写长期记忆
        search_memory,  # 读长期记忆
        # ... 其他工具
    ]
    # 3. 创建 Agent
    llm = ChatDeepSeek(model="deepseek-chat")

    # ── 创建 Agent ─────────────────────────────────────────────────────
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "你是天宫医疗的智能助手。"
            "当用户提到重要的个人信息或病史时，使用 save_memory 工具记住它。"
            "当需要回忆用户历史信息时，使用 search_memory 工具检索。"
        ),
        middleware=[
            SummarizationMiddleware(
                model="deepseek-chat",
                trigger=[
                    ("tokens", 4000),  # token数达到4k时触发
                    ("messages", 6)  # 或消息数达到 4条时触发
                ],
                keep=("messages", 6),  # 摘要后保留最近 4 条消息
            )
        ],
        checkpointer=checkpointer, # 短期记忆
        store=store,  # 长期记忆
    )
    return agent


# 模块级单例：避免每次请求都重新创建 agent 和 checkpointer
_supervisor_agent = None


async def get_supervisor_agent():
    """返回全局单例 Agent，首次调用时初始化。"""
    global _supervisor_agent
    if _supervisor_agent is None:
        _supervisor_agent = await create_supervisor_agent()
    return _supervisor_agent


# FastAPI 路由中使用
async def chat_endpoint(user_id: str, session_id: str, message: str):
    agent = await get_supervisor_agent()  # 使用单例，不重复初始化

    config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return result["messages"][-1].content