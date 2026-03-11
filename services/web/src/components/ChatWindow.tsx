"use client";

import { useState, useRef, useEffect } from "react";
import {
  ChatMessage,
  PrecisionSettings,
  sendMessage,
  createConversation,
} from "@/lib/api";
import { Theme } from "@/lib/useTheme";
import DeliberationPanel from "./DeliberationPanel";
import MixingBoard from "./MixingBoard";

interface ChatWindowProps {
  precision: PrecisionSettings;
  onPrecisionChange: (precision: PrecisionSettings) => void;
  theme: Theme;
  conversationId: string | null;
  messages: ChatMessage[];
  onConversationCreated: (id: string) => void;
  onMessagesChanged: (messages: ChatMessage[]) => void;
}

export default function ChatWindow({
  precision,
  onPrecisionChange,
  theme,
  conversationId,
  messages,
  onConversationCreated,
  onMessagesChanged,
}: ChatWindowProps) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mixingBoardOpen, setMixingBoardOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!mixingBoardOpen) return;
    function handleClick(e: MouseEvent) {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        setMixingBoardOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [mixingBoardOpen]);

  async function ensureConversation(): Promise<string> {
    if (conversationId) return conversationId;
    const conv = await createConversation();
    onConversationCreated(conv.id);
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
    const withUser = [...messages, userMessage];
    onMessagesChanged(withUser);
    setLoading(true);

    try {
      const convId = await ensureConversation();
      const response = await sendMessage(convId, userContent, precision, true);
      onMessagesChanged([...withUser, response.message]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  const { nameGradientLight: gl, nameGradientMid: gm, nameGradientDark: gd } = theme;
  const nameGradient = `linear-gradient(90deg, ${gl}, ${gm}, ${gd}, ${gm}, ${gl})`;

  const t = "color 0.4s ease, background-color 0.4s ease, border-color 0.4s ease, border-radius 0.4s ease, letter-spacing 0.4s ease, font-variation-settings 0.4s ease";

  return (
    <div style={styles.container}>
      <div style={styles.messages}>
        {messages.length === 0 && (
          <div style={styles.empty}>
            <style>{`
              @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
              }
            `}</style>
            <div style={{
              ...styles.greeting,
              fontFamily: theme.fontFamily,
              fontVariationSettings: theme.fontVariationSettings,
              letterSpacing: theme.letterSpacing,
              transition: `${t}, opacity 0.3s ease`,
            }}>
              <span style={{ color: theme.accent, transition: t }}>Hi, </span>
              <span style={{
                display: "inline-block",
                backgroundImage: nameGradient,
                backgroundSize: "200% 100%",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
                animation: "gradientShift 4s ease infinite",
              }}>Mehul</span>
            </div>
            <p style={{
              ...styles.greetingSub,
              color: theme.textMuted,
              fontFamily: theme.fontFamily,
              fontVariationSettings: theme.fontVariationSettings,
              transition: t,
            }}>
              What do you want to talk about today, bud?
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              ...styles.message,
              borderRadius: theme.messageBorderRadius,
              transition: t,
              ...(msg.role === "user"
                ? { alignSelf: "flex-end" as const, backgroundColor: theme.accent, color: "white" }
                : { alignSelf: "flex-start" as const, backgroundColor: theme.surfaceBg, color: theme.text }
              ),
            }}
          >
            <div style={styles.messageRole}>
              {msg.role === "user" ? "You" : "empath"}
            </div>
            <div style={{
              ...styles.messageContent,
              fontFamily: theme.fontFamily,
              fontVariationSettings: theme.fontVariationSettings,
            }}>
              {msg.content}
            </div>
            {msg.deliberation && (
              <DeliberationPanel deliberation={msg.deliberation} />
            )}
          </div>
        ))}

        {loading && (
          <div style={{
            ...styles.message,
            borderRadius: theme.messageBorderRadius,
            alignSelf: "flex-start" as const,
            backgroundColor: theme.surfaceBg,
            color: theme.text,
          }}>
            <div style={styles.messageRole}>empath</div>
            <div style={{ ...styles.thinking, color: theme.textMuted }}>
              Deliberating...
            </div>
          </div>
        )}

        {error && <div style={styles.error}>{error}</div>}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div style={{
        ...styles.inputWrapper,
        borderTop: `1px solid ${theme.border}`,
        transition: t,
      }}>
        {mixingBoardOpen && (
          <div ref={popupRef} style={{
            ...styles.popup,
            backgroundColor: theme.sidebarBg,
            border: `1px solid ${theme.border}`,
            borderRadius: theme.borderRadius,
          }}>
            <MixingBoard precision={precision} onChange={onPrecisionChange} />
          </div>
        )}

        <form onSubmit={handleSubmit} style={styles.inputArea}>
          <button
            type="button"
            onClick={() => setMixingBoardOpen(!mixingBoardOpen)}
            style={{
              ...styles.mixBtn,
              backgroundColor: mixingBoardOpen ? theme.accent : theme.surfaceBg,
              borderColor: mixingBoardOpen ? theme.accent : theme.border,
              color: mixingBoardOpen ? "#fff" : theme.textMuted,
              borderRadius: theme.borderRadius,
              transition: t,
            }}
            title="Mixing board"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="4" y1="21" x2="4" y2="14" />
              <line x1="4" y1="10" x2="4" y2="3" />
              <line x1="12" y1="21" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12" y2="3" />
              <line x1="20" y1="21" x2="20" y2="16" />
              <line x1="20" y1="12" x2="20" y2="3" />
              <line x1="1" y1="14" x2="7" y2="14" />
              <line x1="9" y1="8" x2="15" y2="8" />
              <line x1="17" y1="16" x2="23" y2="16" />
            </svg>
          </button>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={theme.placeholder}
            disabled={loading}
            style={{
              ...styles.input,
              backgroundColor: theme.surfaceBg,
              border: `1px solid ${theme.border}`,
              color: theme.text,
              borderRadius: theme.borderRadius,
              fontFamily: theme.fontFamily,
              fontVariationSettings: theme.fontVariationSettings,
              transition: t,
            }}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            style={{
              ...styles.button,
              backgroundColor: theme.accent,
              borderRadius: theme.borderRadius,
              transition: t,
            }}
          >
            Send
          </button>
        </form>
      </div>
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
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    flex: 1,
    maxWidth: "560px",
    width: "100%",
    margin: "0 auto",
  },
  greeting: {
    fontSize: "72px",
    fontWeight: 700,
    lineHeight: 1.2,
  },
  greetingSub: {
    marginTop: "16px",
    fontSize: "15px",
  },
  message: {
    padding: "12px 16px",
    maxWidth: "80%",
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
    fontStyle: "italic",
  },
  error: {
    alignSelf: "center",
    color: "#f87171",
    fontSize: "13px",
    padding: "8px",
  },
  inputWrapper: {
    position: "relative",
  },
  popup: {
    position: "absolute",
    bottom: "100%",
    left: "16px",
    marginBottom: "8px",
    width: "260px",
    boxShadow: "0 -4px 20px rgba(0,0,0,0.4)",
    zIndex: 20,
    overflow: "hidden",
  },
  inputArea: {
    display: "flex",
    gap: "8px",
    padding: "16px",
    alignItems: "center",
  },
  mixBtn: {
    width: "36px",
    height: "36px",
    borderStyle: "solid",
    borderWidth: "1px",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    fontSize: "14px",
    outline: "none",
  },
  button: {
    padding: "10px 20px",
    fontSize: "14px",
    color: "white",
    border: "none",
    cursor: "pointer",
    fontWeight: 500,
  },
};
