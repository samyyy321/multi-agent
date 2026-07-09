import type { ChatMessage } from "../types/chat";

export type ChatStatus = "idle" | "streaming" | "error";

export interface ChatState {
  messages: ChatMessage[];
  status: ChatStatus;
  errorMessage: string | null;
}

export type ChatAction =
  | { type: "start"; userContent: string; assistantId: string }
  | { type: "appendToken"; content: string }
  | { type: "complete" }
  | { type: "fail"; message: string }
  | { type: "reset" };

/** 创建没有历史消息的初始聊天状态。 */
export function createInitialChatState(): ChatState {
  return {
    messages: [],
    status: "idle",
    errorMessage: null,
  };
}

/**
 * 聊天状态机集中管理流式文本拼接，避免组件各自修改消息数组。
 * 每轮请求始终新增一个空助手消息，token 仅追加给该消息。
 */
export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "start":
      return {
        messages: [
          ...state.messages,
          {
            id: `user-${action.assistantId}`,
            role: "user",
            content: action.userContent,
          },
          {
            id: action.assistantId,
            role: "assistant",
            content: "",
          },
        ],
        status: "streaming",
        errorMessage: null,
      };
    case "appendToken": {
      const lastMessage = state.messages.at(-1);
      if (!lastMessage || lastMessage.role !== "assistant") {
        return state;
      }
      return {
        ...state,
        messages: [
          ...state.messages.slice(0, -1),
          { ...lastMessage, content: lastMessage.content + action.content },
        ],
      };
    }
    case "complete":
      return { ...state, status: "idle", errorMessage: null };
    case "fail":
      return { ...state, status: "error", errorMessage: action.message };
    case "reset":
      return createInitialChatState();
  }
}