# 医疗多智能体

面向医疗咨询场景的多智能体服务，提供智慧问诊、知识文档检索、医学知识图谱检索与会话记忆能力。

> 项目用于辅助问诊和知识检索，不替代医生诊断、处方或急救处置。

## 核心能力

- **智慧问诊**：患者上下文、急症识别、症状标准化、候选疾病、追问、LLM 结论审查和问诊记录保存。
- **知识问答**：文档 RAG、知识图谱 RAG、NL2SQL 与多通道融合检索。
- **多智能体编排**：Supervisor 根据会话状态调度问诊与知识问答 Worker。
- **记忆能力**：Redis 保存会话级状态，Milvus 保存长期语义记忆与知识文档向量。
- **Docker 基础设施**：PostgreSQL、Redis Stack、Milvus、Neo4j、MinIO 与 Attu。

## 文档

完整的架构、配置、数据初始化、文档入库、API、测试和排障说明见：

- [项目说明书](docs/项目说明书.md)

## 快速开始

### 1. 激活 Conda 环境并安装依赖

```powershell
conda activate multi-agent
python -m pip install -r requirements.txt
```

### 2. 启动基础设施

```powershell
docker compose up -d
docker compose ps
```

### 3. 配置环境变量

```powershell
Copy-Item .env.example .env
```

在 `.env` 中至少配置可用的 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`CHAT_MODEL` 与 `EMBEDDING_MODEL`。

### 4. 初始化开发数据

```powershell
alembic upgrade head
python scripts/init_postgres.py
python scripts/init_neo4j.py
python scripts/init_symptom_index.py
```

> `init_neo4j.py` 会重建 Neo4j 图数据；已有图谱数据时请谨慎执行。

### 5. 启动 API

```powershell
python -m uvicorn src.main:app --port 8080 --reload
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/health/deps | ConvertTo-Json -Depth 5
```

## 导入知识文档

把待导入的 PDF、Word、Markdown 或文本文件放入：

```text
scripts/data/
```

执行：

```powershell
python scripts/import_knowledge_doc.py --file 高血压指南.pdf --doc-type guideline --category 心血管
```

查看所有参数：

```powershell
python scripts/import_knowledge_doc.py --help
```

## API 入口

| 接口 | 说明 |
| --- | --- |
| `GET /health` | 应用基础健康检查 |
| `GET /health/deps` | PostgreSQL、Redis、MinIO、Milvus、Neo4j 健康检查 |
| `POST /api/v1/chat` | 非流式聊天接口 |
| `POST /api/v1/chat/stream` | SSE 流式聊天接口 |

非流式请求示例：

```json
{
  "user_id": "test-user-001",
  "session_id": "inquiry-001",
  "message": "我头痛发热两天了，体温38.5度，应该挂哪个科？",
  "patient_id": 42
}
```

## 常用测试

```powershell
python -m pytest test/test_inquiry_candidates.py -q
python -m pytest test/test_inquiry_conclusion.py -q
python -m pytest test/test_import_knowledge_doc.py -q
```

## 开发注意事项

- `patient_id` 必须存在于 PostgreSQL `patients` 表。开发初始化脚本会创建 `patient_id=42` 的测试患者。
- 文档解析可本地回退到 LlamaIndex，但文本向量化与查询仍依赖 DashScope Embedding 网络。
- 涉及急症信息时，系统会优先提示急诊处理；不要将普通问诊结论当作确定诊断。
