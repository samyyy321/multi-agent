import { describe, expect, test } from "vitest";

import { chatReducer, createInitialChatState } from "./state";

describe("chatReducer", () => {
  test("token 追加到当前助手消息", () => {
    const state = createInitialChatState();
    const afterStart = chatReducer(state, {
      type: "start",
      userContent: "我发热了",
      assistantId: "a-1",
    });
    const afterToken = chatReducer(afterStart, {
      type: "appendToken",
      content: "请问体温多少？",
    });

    expect(afterToken.status).toBe("streaming");
    expect(afterToken.messages.at(-1)).toMatchObject({
      id: "a-1",
      role: "assistant",
      content: "请问体温多少？",
    });
  });

  test("错误时保留已生成的助手内容", () => {
    const state = createInitialChatState();
    const started = chatReducer(state, {
      type: "start",
      userContent: "我头痛",
      assistantId: "a-2",
    });
    const partial = chatReducer(started, { type: "appendToken", content: "建议" });
    const failed = chatReducer(partial, { type: "fail", message: "暂时无法连接问诊服务，请稍后重试。" });

    expect(failed.status).toBe("error");
    expect(failed.errorMessage).toBe("暂时无法连接问诊服务，请稍后重试。");
    expect(failed.messages.at(-1)?.content).toBe("建议");
  });

  test("新会话移除消息并恢复空闲状态", () => {
    const started = chatReducer(createInitialChatState(), {
      type: "start",
      userContent: "我咳嗽",
      assistantId: "a-3",
    });
    const reset = chatReducer(started, { type: "reset" });

    expect(reset).toEqual({
      messages: [],
      status: "idle",
      errorMessage: null,
    });
  });
});