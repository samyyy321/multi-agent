import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { streamChat } from "../api/chat";
import { chatReducer, createInitialChatState } from "../chat/state";
import type { ChatRequest } from "../types/chat";

const USER_ID_STORAGE_KEY = "medical-chat-user-id";
const CONNECTION_ERROR_MESSAGE = "暂时无法连接问诊服务，请稍后重试。";

function createId(): string {
  return crypto.randomUUID();
}

/** 读取或首次生成浏览器本地用户标识，不把它当作认证身份。 */
function getBrowserUserId(): string {
  const storedId = window.localStorage.getItem(USER_ID_STORAGE_KEY);
  if (storedId) {
    return storedId;
  }

  const userId = createId();
  window.localStorage.setItem(USER_ID_STORAGE_KEY, userId);
  return userId;
}

function toPatientId(value: string): number | undefined {
  const trimmedValue = value.trim();
  return /^\d+$/.test(trimmedValue) ? Number(trimmedValue) : undefined;
}

/**
 * 管理患者聊天会话和单个流式请求。
 * 同一时刻只允许一个请求，避免多个 SSE 流交叉写入消息列表。
 */
export function useChat() {
  const [state, dispatch] = useReducer(chatReducer, undefined, createInitialChatState);
  const [userId] = useState(getBrowserUserId);
  const [sessionId, setSessionId] = useState(createId);
  const [patientId, setPatientId] = useState("");
  const abortControllerRef = useRef<AbortController | null>(null);

  const stopGenerating = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    dispatch({ type: "complete" });
  }, []);

  const startNewSession = useCallback(() => {
    stopGenerating();
    setSessionId(createId());
    dispatch({ type: "reset" });
  }, [stopGenerating]);

  const sendMessage = useCallback(
    async (content: string) => {
      const message = content.trim();
      if (!message || state.status === "streaming") {
        return;
      }

      const assistantId = createId();
      const controller = new AbortController();
      const selectedPatientId = toPatientId(patientId);
      const request: ChatRequest = {
        user_id: userId,
        session_id: sessionId,
        message,
        ...(selectedPatientId === undefined ? {} : { patient_id: selectedPatientId }),
      };

      abortControllerRef.current = controller;
      dispatch({ type: "start", userContent: message, assistantId });
      let receivedServerError = false;

      try {
        await streamChat(request, {
          signal: controller.signal,
          onEvent: (event) => {
            if (event.type === "token") {
              dispatch({ type: "appendToken", content: event.content });
            } else if (event.type === "done") {
              dispatch({ type: "complete" });
            } else {
              receivedServerError = true;
              dispatch({ type: "fail", message: CONNECTION_ERROR_MESSAGE });
            }
          },
        });

        if (!receivedServerError) {
          dispatch({ type: "complete" });
        }
      } catch (error) {
        const aborted = controller.signal.aborted || (error instanceof DOMException && error.name === "AbortError");
        if (!aborted) {
          dispatch({ type: "fail", message: CONNECTION_ERROR_MESSAGE });
        }
      } finally {
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
      }
    },
    [patientId, sessionId, state.status, userId],
  );

  useEffect(() => () => abortControllerRef.current?.abort(), []);

  return {
    state,
    userId,
    sessionId,
    patientId,
    setPatientId,
    sendMessage,
    stopGenerating,
    startNewSession,
  };
}