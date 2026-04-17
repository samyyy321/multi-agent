# 在 Python 交互环境或测试脚本中运行
import asyncio
from src.agents.workers.inquiry_agent import get_inquiry_agent

async def test_inquiry():
    agent = get_inquiry_agent()
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "我头疼发烧两天了，体温38.5度"}]
    })
    print(result["messages"][-1].content)