import { useEffect, useRef } from "react";

import type { ChatMessage } from "../types/chat";

const SUGGESTED_QUESTIONS = [
  "我头痛发热两天了，应该挂哪个科？",
  "咳嗽、喉咙痛应该注意什么？",
  "胃痛反酸时需要尽快就医吗？",
];

interface MessageListProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  onSuggestion: (question: string) => void;
}

/** 消息列表负责展示对话内容和初次进入页面时的引导问题。 */
export function MessageList({ messages, isStreaming, onSuggestion }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isStreaming]);

  if (messages.length === 0) {
    return (
      <section className="empty-chat" aria-label="智能问诊开始页">
        <div className="assistant-avatar large" aria-hidden="true">医</div>
        <span className="eyebrow">sam医疗智能问诊</span>
        <h1>您好，有什么可以帮您？</h1>
        <p>请尽量描述不适部位、持续时间、体温以及伴随症状，我会协助您判断下一步就诊方向。</p>
        <div className="suggestion-list" aria-label="快捷提问">
          {SUGGESTED_QUESTIONS.map((question) => (
            <button key={question} type="button" className="suggestion-button" onClick={() => onSuggestion(question)}>
              {question}
              <span aria-hidden="true">→</span>
            </button>
          ))}
        </div>
      </section>
    );
  }

  return (
    <section className="message-list" aria-live="polite" aria-relevant="additions text" aria-label="问诊对话">
      {messages.map((message) => {
        const isAssistant = message.role === "assistant";
        const waiting = isAssistant && isStreaming && !message.content;
        return (
          <article key={message.id} className={`message-row ${message.role}`}>
            {isAssistant && <div className="assistant-avatar" aria-hidden="true">医</div>}
            <div className={`message-bubble ${message.role}`}>
              <span className="message-author">{isAssistant ? "sam医疗助手" : "您"}</span>
              {waiting ? (
                <span className="typing-indicator" aria-label="正在生成回复">
                  <i />
                  <i />
                  <i />
                  <em>正在生成回复</em>
                </span>
              ) : (
                <p>{message.content}</p>
              )}
            </div>
          </article>
        );
      })}
      <div ref={bottomRef} />
    </section>
  );
}