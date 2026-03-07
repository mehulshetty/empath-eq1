"use client";

import { useState, useRef, useEffect } from "react";
import {
  ChatMessage,
  PrecisionSettings,
  sendMessage,
  createConversation,
} from "@/lib/api";
import DeliberationPanel from "./DeliberationPanel";

interface ChatWindowProps {
  precision: PrecisionSettings;
}

/**
 * Main chat interface. Manages a conversation, sends messages to
 * the API, and displays the response stream with optional
 * deliberation metadata.
 */
export default function ChatWindow({ precision }: ChatWindowProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function ensureConversation(): Promise<string> {
    if (conversationId) return conversationId;
    const conv = await createConversation();
    setConversationId(conv.id);
    return conv.id;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userContent = input.trim();
    setInput("");
    setError(null);

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: userContent,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const convId = await ensureConversation();
      const response = await sendMessage(convId, userContent, precision, true);
      setMessages((prev) => [...prev, response.message]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.container}>
      {/* Messages area */}
      <div style={styles.messages}>
        {messages.length === 0 && (
          <div style={styles.empty}>
            Send a message to start a conversation with CLSA.
            <br />
            Adjust the mixing board to control Logic and EQ emphasis.
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              ...styles.message,
              ...(msg.role === "user" ? styles.userMessage : styles.assistantMessage),
            }}
          >
            <div style={styles.messageRole}>
              {msg.role === "user" ? "You" : "CLSA"}
            </div>
            <div style={styles.messageContent}>{msg.content}</div>
            {msg.deliberation && (
              <DeliberationPanel deliberation={msg.deliberation} />
            )}
          </div>
        ))}

        {loading && (
          <div style={{ ...styles.message, ...styles.assistantMessage }}>
            <div style={styles.messageRole}>CLSA</div>
            <div style={styles.thinking}>Deliberating...</div>
          </div>
        )}

        {error && <div style={styles.error}>{error}</div>}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <form onSubmit={handleSubmit} style={styles.inputArea}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          disabled={loading}
          style={styles.input}
        />
        <button type="submit" disabled={loading || !input.trim()} style={styles.button}>
          Send
        </button>
      </form>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
    flex: 1,
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    padding: "16px",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  empty: {
    textAlign: "center",
    color: "#888",
    marginTop: "40%",
    lineHeight: 1.6,
  },
  message: {
    padding: "12px 16px",
    borderRadius: "8px",
    maxWidth: "80%",
  },
  userMessage: {
    alignSelf: "flex-end",
    backgroundColor: "#0066cc",
    color: "white",
  },
  assistantMessage: {
    alignSelf: "flex-start",
    backgroundColor: "#f0f0f0",
    color: "#222",
  },
  messageRole: {
    fontSize: "11px",
    fontWeight: 600,
    marginBottom: "4px",
    opacity: 0.7,
  },
  messageContent: {
    fontSize: "14px",
    lineHeight: 1.5,
    whiteSpace: "pre-wrap",
  },
  thinking: {
    fontSize: "14px",
    color: "#888",
    fontStyle: "italic",
  },
  error: {
    alignSelf: "center",
    color: "#cc0000",
    fontSize: "13px",
    padding: "8px",
  },
  inputArea: {
    display: "flex",
    gap: "8px",
    padding: "16px",
    borderTop: "1px solid #e0e0e0",
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    fontSize: "14px",
    border: "1px solid #ddd",
    borderRadius: "6px",
    outline: "none",
  },
  button: {
    padding: "10px 20px",
    fontSize: "14px",
    backgroundColor: "#0066cc",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontWeight: 500,
  },
};
