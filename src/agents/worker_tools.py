# src/agents/worker_tools.py

from langchain_core.tools import tool
from src.agents.workers.inquiry_agent import get_inquiry_agent
from src.agents.workers.knowledge_agent import get_knowledge_agent


@tool
async def call_inquiry_agent(message: str) -> str:
    """
    调用智慧问诊Agent，对患者进行智能分诊。
    适用场景：患者描述症状、询问挂哪个科室、需要预约挂号时。
    message: 患者描述的症状或问诊需求。
    """
    agent = get_inquiry_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_knowledge_agent(message: str) -> str:
    """
    调用知识问答Agent，回答医学知识类问题。
    适用场景：询问疾病知识、治疗方案、医学术语解释、文献检索时。
    message: 患者或医生的医学知识问题。
    """
    agent = get_knowledge_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


# 所有 Worker 工具列表，供 Supervisor 使用
WORKER_TOOLS = [
    call_inquiry_agent,
    call_knowledge_agent
]