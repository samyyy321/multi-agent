import type { ChatRequest, ChatStreamEvent } from "../types/chat";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8080";
const STREAM_PATH = "/api/v1/chat/stream";
const UNRECOGNIZED_EVENT_MESSAGE = "服务返回了无法识别的数据";
const CONNECTION_ERROR_MESSAGE = "暂时无法连接问诊服务，请稍后重试。";

function getApiBaseUrl(): string {
  const configuredUrl = import.meta.env.VITE_API_BASE_URL?.trim();
  return (configuredUrl || DEFAULT_API_BASE_URL).replace(/\/$/, "");
}

function isStreamEvent(value: unknown): value is ChatStreamEvent {
  if (!value || typeof value !== "object" || !("type" in value)) {
    return false;
  }

  const event = value as Record<string, unknown>;
  if (event.type === "token") {
    return typeof event.content === "string";
  }
  if (event.type === "done") {
    return typeof event.session_id === "string";
  }
  if (event.type === "error") {
    return typeof event.message === "string";
  }
  return false;
}

/**
 * 解析单个 SSE 帧，只接受后端约定的 token、done、error 三种事件。
 * 空帧用于分隔数据块，不产生界面状态更新。
 */
export function parseSseFrame(frame: string): ChatStreamEvent | null {
  const data = frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");

  if (!data) {
    return null;
  }

  try {
    const event: unknown = JSON.parse(data);
    if (!isStreamEvent(event)) {
      throw new Error(UNRECOGNIZED_EVENT_MESSAGE);
    }
    return event;
  } catch {
    throw new Error(UNRECOGNIZED_EVENT_MESSAGE);
  }
}

/** 使用 fetch 读取 POST SSE 响应，支持 AbortController 取消请求。 */
export async function streamChat(
  request: ChatRequest,
  options: {
    signal: AbortSignal;
    onEvent: (event: ChatStreamEvent) => void;
  },
): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}${STREAM_PATH}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal: options.signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(CONNECTION_ERROR_MESSAGE);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pendingText = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      pendingText += decoder.decode(value, { stream: !done });

      const frames = pendingText.split(/\r?\n\r?\n/);
      pendingText = frames.pop() ?? "";
      for (const frame of frames) {
        const event = parseSseFrame(frame);
        if (event) {
          options.onEvent(event);
        }
      }

      if (done) {
        break;
      }
    }

    if (pendingText.trim()) {
      const event = parseSseFrame(pendingText);
      if (event) {
        options.onEvent(event);
      }
    }
  } finally {
    reader.releaseLock();
  }
}