import { useState, type KeyboardEvent } from "react";

interface ChatComposerProps {
  disabled: boolean;
  isStreaming: boolean;
  onSend: (content: string) => void;
  onStop: () => void;
}

/** 输入区统一处理发送、换行和停止生成交互。 */
export function ChatComposer({ disabled, isStreaming, onSend, onStop }: ChatComposerProps) {
  const [content, setContent] = useState("");
  const canSend = content.trim().length > 0 && !disabled;

  const send = () => {
    if (!canSend) {
      return;
    }
    onSend(content);
    setContent("");
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  };

  return (
    <div className="composer-wrap">
      <div className="composer">
        <label className="sr-only" htmlFor="chat-message">请输入您的症状或问题</label>
        <textarea
          id="chat-message"
          rows={1}
          placeholder="描述您的症状或想咨询的问题…"
          value={content}
          disabled={disabled}
          onChange={(event) => setContent(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        {isStreaming ? (
          <button type="button" className="stop-button" onClick={onStop} aria-label="停止生成回复">
            <span aria-hidden="true">■</span>
            停止
          </button>
        ) : (
          <button type="button" className="send-button" onClick={send} disabled={!canSend} aria-label="发送消息">
            <span aria-hidden="true">↑</span>
          </button>
        )}
      </div>
      <p className="composer-hint">Enter 发送，Shift + Enter 换行</p>
    </div>
  );
}