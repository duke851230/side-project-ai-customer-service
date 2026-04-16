import { useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const SESSION_ID_KEY = "chat_session_id";
const WELCOME_MESSAGE = {
  role: "assistant",
  text: "你好，我可以幫你查訂單狀態或回答 FAQ。",
  route: "welcome",
  citations: [],
};

const createSessionId = () => (
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID().replace(/-/g, "")
    : `${Date.now()}${Math.random().toString(16).slice(2)}`
);

const getOrCreateSessionId = () => {
  const existing = window.localStorage.getItem(SESSION_ID_KEY);
  if (existing) return existing;

  const created = createSessionId();
  window.localStorage.setItem(SESSION_ID_KEY, created);
  return created;
};

function App() {
  const [message, setMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState(() => getOrCreateSessionId());
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);

  const canSend = useMemo(() => message.trim().length > 0 && !isSending, [message, isSending]);

  const sendMessage = async () => {
    const trimmed = message.trim();
    if (!trimmed || isSending) return;

    setError("");
    setIsSending(true);
    setMessage("");
    setMessages((prev) => [...prev, { role: "user", text: trimmed }]);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, session_id: sessionId }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      if (typeof data.session_id === "string" && data.session_id.trim()) {
        setSessionId(data.session_id);
        window.localStorage.setItem(SESSION_ID_KEY, data.session_id);
      }
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer || "(empty)",
          route: data.route || "unknown",
          citations: Array.isArray(data.citations) ? data.citations : [],
        },
      ]);
    } catch (err) {
      setError(`訊息送出失敗：${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setIsSending(false);
    }
  };

  const startNewConversation = () => {
    if (isSending) return;
    const newSessionId = createSessionId();
    setSessionId(newSessionId);
    window.localStorage.setItem(SESSION_ID_KEY, newSessionId);
    setMessage("");
    setError("");
    setMessages([WELCOME_MESSAGE]);
  };

  return (
    <main className="page">
      <section className="chat-card">
        <header className="chat-header">
          <h1>AI Customer Service</h1>
          <p>Minimal React Chat UI</p>
          <button type="button" onClick={startNewConversation} disabled={isSending}>
            開始新對話
          </button>
        </header>

        <div className="messages" aria-live="polite">
          {messages.map((item, index) => (
            <article key={`${item.role}-${index}`} className={`bubble ${item.role}`}>
              <p>{item.text}</p>
              {item.role === "assistant" && (
                <div className="meta">
                  <span>route: {item.route}</span>
                  {item.citations?.length > 0 && (
                    <ul>
                      {item.citations.map((citation) => (
                        <li key={citation}>{citation}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </article>
          ))}
        </div>

        {error && <p className="error">{error}</p>}

        <footer className="composer">
          <textarea
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
              }
            }}
            placeholder="輸入問題，Enter 送出，Shift+Enter 換行"
            rows={2}
          />
          <button type="button" onClick={sendMessage} disabled={!canSend}>
            {isSending ? "Sending..." : "Send"}
          </button>
        </footer>
      </section>
    </main>
  );
}

export default App;
