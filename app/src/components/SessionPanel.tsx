interface SessionPanelProps {
  userId: string;
  sessionId: string;
  patientId: string;
  onPatientIdChange: (value: string) => void;
}

function shortenId(value: string): string {
  return `${value.slice(0, 8)}…${value.slice(-4)}`;
}

/** 展示当前浏览器会话信息和患者侧使用提示。 */
export function SessionPanel({
  userId,
  sessionId,
  patientId,
  onPatientIdChange,
}: SessionPanelProps) {
  return (
    <aside className="session-panel" aria-label="问诊会话设置">
      <div className="panel-heading">
        <span className="eyebrow">当前咨询</span>
        <h2>就诊信息</h2>
        <p>填写信息可帮助系统在开发联调环境中读取对应病史。</p>
      </div>

      <label className="field-label" htmlFor="patient-id">
        患者档案 ID（可选）
      </label>
      <input
        id="patient-id"
        className="patient-input"
        inputMode="numeric"
        placeholder="例如：42"
        value={patientId}
        onChange={(event) => onPatientIdChange(event.target.value)}
      />
      <p className="field-hint">开发联调可填写 42。该字段不是登录凭据或身份认证。</p>

      <dl className="session-details">
        <div>
          <dt>浏览器标识</dt>
          <dd title={userId}>{shortenId(userId)}</dd>
        </div>
        <div>
          <dt>本次会话</dt>
          <dd title={sessionId}>{shortenId(sessionId)}</dd>
        </div>
      </dl>

      <section className="safety-card" aria-labelledby="safety-title">
        <span className="safety-icon" aria-hidden="true">!</span>
        <div>
          <h3 id="safety-title">医疗安全提示</h3>
          <p>胸痛、呼吸困难、意识障碍等紧急情况，请立即前往急诊或拨打当地急救电话。</p>
        </div>
      </section>
    </aside>
  );
}