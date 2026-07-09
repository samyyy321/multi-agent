import { ChatComposer } from "./components/ChatComposer";
import { MessageList } from "./components/MessageList";
import { SessionPanel } from "./components/SessionPanel";
import { useChat } from "./hooks/useChat";

/** 患者侧智能问诊页面，组合会话信息、对话内容和输入控制。 */
function App() {
  const {
    state,
    userId,
    sessionId,
    patientId,
    setPatientId,
    sendMessage,
    stopGenerating,
    startNewSession,
  } = useChat();
  const isStreaming = state.status === "streaming";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="sam医疗智能问诊首页">
          <span className="brand-mark" aria-hidden="true">+</span>
          <span>sam医疗</span>
          <small>智能问诊</small>
        </a>
        <div className="topbar-actions">
          <span className="service-status"><i aria-hidden="true" />服务已就绪</span>
          <button type="button" className="new-session-button" onClick={startNewSession}>
            <span aria-hidden="true">＋</span>
            新建会话
          </button>
        </div>
      </header>

      <div className="app-layout">
        <SessionPanel
          userId={userId}
          sessionId={sessionId}
          patientId={patientId}
          onPatientIdChange={setPatientId}
        />

        <main className="chat-workspace">
          <div className="workspace-heading">
            <div>
              <span className="eyebrow">在线医疗咨询</span>
              <h1>智能问诊助手</h1>
            </div>
            <span className="privacy-note">对话仅在当前页面展示</span>
          </div>

          {state.errorMessage && (
            <div className="error-banner" role="alert">
              <span aria-hidden="true">!</span>
              <p>{state.errorMessage}</p>
            </div>
          )}

          <MessageList messages={state.messages} isStreaming={isStreaming} onSuggestion={sendMessage} />
          <ChatComposer
            disabled={isStreaming}
            isStreaming={isStreaming}
            onSend={sendMessage}
            onStop={stopGenerating}
          />
        </main>
      </div>

      <footer className="page-footer">
        <span aria-hidden="true">△</span>
        仅供辅助咨询，不替代医生诊断或紧急处置。
      </footer>
    </div>
  );
}

export default App;