/** 发送给医疗聊天后端的请求数据。 */
export interface ChatRequest {
  user_id: string;
  session_id: string;
  message: string;
  patient_id?: number;
}

/** 聊天页面中展示的一条消息。 */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

/** 后端 SSE 协议支持的事件类型。 */
export type ChatStreamEvent =
  | { type: "token"; content: string }
  | { type: "done"; session_id: string }
  | { type: "error"; message: string };