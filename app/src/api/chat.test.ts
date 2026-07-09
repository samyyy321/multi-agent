import { afterEach, describe, expect, test, vi } from "vitest";

import { parseSseFrame, streamChat } from "./chat";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("parseSseFrame", () => {
  test("解析 token 事件", () => {
    expect(parseSseFrame('data: {"type":"token","content":"你好"}')).toEqual({
      type: "token",
      content: "你好",
    });
  });

  test("解析 done 和 error 事件", () => {
    expect(parseSseFrame('data: {"type":"done","session_id":"s-1"}')).toEqual({
      type: "done",
      session_id: "s-1",
    });
    expect(parseSseFrame('data: {"type":"error","message":"服务繁忙"}')).toEqual({
      type: "error",
      message: "服务繁忙",
    });
  });

  test("忽略空帧并拒绝非法事件", () => {
    expect(parseSseFrame("\n")).toBeNull();
    expect(() => parseSseFrame("data: not-json")).toThrow("服务返回了无法识别的数据");
    expect(() => parseSseFrame('data: {"type":"other"}')).toThrow(
      "服务返回了无法识别的数据",
    );
  });
});

test("跨数据块拼接后按顺序派发事件", async () => {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode('data: {"type":"token","content":"你'));
      controller.enqueue(
        encoder.encode('好"}\n\ndata: {"type":"done","session_id":"s-1"}\n\n'),
      );
      controller.close();
    },
  });
  const fetchMock = vi.fn().mockResolvedValue(new Response(stream, { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  const events: unknown[] = [];

  await streamChat(
    {
      user_id: "user-1",
      session_id: "s-1",
      message: "你好",
      patient_id: 42,
    },
    {
      signal: new AbortController().signal,
      onEvent: (event) => events.push(event),
    },
  );

  expect(events).toEqual([
    { type: "token", content: "你好" },
    { type: "done", session_id: "s-1" },
  ]);
  expect(fetchMock).toHaveBeenCalledWith(
    "http://127.0.0.1:8080/api/v1/chat/stream",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        user_id: "user-1",
        session_id: "s-1",
        message: "你好",
        patient_id: 42,
      }),
    }),
  );
});

test("非成功响应显示通用错误", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 503 })));

  await expect(
    streamChat(
      { user_id: "user-1", session_id: "s-1", message: "你好" },
      { signal: new AbortController().signal, onEvent: vi.fn() },
    ),
  ).rejects.toThrow("暂时无法连接问诊服务，请稍后重试。");
});