# sam医疗智能问诊前端

面向患者的单页聊天界面，通过现有 FastAPI 流式接口提供智能问诊与医疗咨询体验。

> 本页面仅提供辅助咨询，不替代医生诊断、处方或紧急处置。出现胸痛、呼吸困难、意识障碍等紧急情况，请立即线下就医或联系当地急救服务。

## 环境要求

- Node.js 20 或更高版本；
- 可访问的后端 API 服务；
- 后端已允许前端开发服务器跨域访问。

## 安装

```powershell
Set-Location app
npm install
```

## 配置

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

`.env` 中可配置：

```text
VITE_API_BASE_URL=http://127.0.0.1:8080
```

未配置时，前端同样默认请求 `http://127.0.0.1:8080`。

## 启动开发服务

```powershell
Set-Location app
npm run dev
```

Vite 默认提供本地开发地址。页面通过 `POST /api/v1/chat/stream` 使用 `fetch` 读取 SSE 流；该接口为 POST，因此不使用只能发起 GET 请求的 `EventSource`。

## 测试与构建

```powershell
Set-Location app
npm run test
npm run build
```

## 使用说明

- 页面首次打开时在浏览器 `localStorage` 中生成并保存 `user_id`；
- 每次点击“新建会话”会生成新的 `session_id`，并清空当前页面消息；
- 聊天正文不写入浏览器长期存储；
- “患者档案 ID”是开发联调输入，不是登录身份或授权凭据；开发环境可填写 `42`；
- 流式请求进行中可以点击“停止”取消，已经生成的文本会保留；
- 发生服务端或网络错误时，页面只显示通用错误提示，不展示后端堆栈内容。

## 联调前提

先启动项目后端与其依赖服务，再启动此前端。后端健康检查可用于确认基础服务是否可达：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

建议用新的会话发起问诊，并在开发环境填写 `patient_id=42` 进行验证。