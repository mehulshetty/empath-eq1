"use client";

import { useState, useEffect, useCallback } from "react";
import ChatWindow from "@/components/ChatWindow";
import {
  PrecisionSettings,
  ConversationSummary,
  ChatMessage,
  listConversations,
  getConversation,
  deleteConversation,
} from "@/lib/api";
import { useTheme } from "@/lib/useTheme";

export default function Home() {
  const [precision, setPrecision] = useState<PrecisionSettings>({
    logic: 1.0,
    eq: 1.0,
  });
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [hoveredConvId, setHoveredConvId] = useState<string | null>(null);
  const theme = useTheme(precision);

  const loadConversations = useCallback(async () => {
    try {
      const convs = await listConversations();
      setConversations(convs);
    } catch {
      // API may not be running yet
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  async function handleSelectConversation(id: string) {
    if (id === activeConvId) return;
    try {
      const detail = await getConversation(id);
      setActiveConvId(id);
      setMessages(detail.messages);
    } catch {
      // ignore
    }
  }

  async function handleNewConversation() {
    setActiveConvId(null);
    setMessages([]);
  }

  async function handleDeleteConversation(id: string) {
    try {
      await deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConvId === id) {
        setActiveConvId(null);
        setMessages([]);
      }
    } catch {
      // ignore
    }
  }

  function handleConversationCreated(id: string) {
    setActiveConvId(id);
    loadConversations();
  }

  function handleMessagesChanged(msgs: ChatMessage[]) {
    setMessages(msgs);
  }

  return (
    <div style={styles.layout}>
      {/* Sidebar */}
      <aside
        style={{
          ...styles.sidebar,
          width: sidebarOpen ? "280px" : "0px",
          overflow: "hidden",
          transition: "width 0.2s ease, background-color 0.4s ease, border-color 0.4s ease",
          backgroundColor: theme.sidebarBg,
          borderRight: `1px solid ${theme.border}`,
        }}
      >
        <div style={{ ...styles.logo, borderBottom: `1px solid ${theme.border}` }}>
          <h1 style={{
            ...styles.logoText,
            fontFamily: theme.fontFamily,
            fontVariationSettings: theme.fontVariationSettings,
            color: theme.text,
            letterSpacing: theme.letterSpacing,
            transition: "color 0.4s ease, font-variation-settings 0.4s ease, letter-spacing 0.4s ease",
          }}>
            empath
          </h1>
          <p style={{ ...styles.logoSubtext, color: theme.textMuted }}>
            Collaborative Latent Superposition Architecture
          </p>
        </div>
        <div style={styles.convList}>
          <button
            onClick={handleNewConversation}
            style={{
              ...styles.newConvBtn,
              backgroundColor: theme.surfaceBg,
              border: `1px solid ${theme.border}`,
              color: theme.text,
              borderRadius: theme.borderRadius,
            }}
          >
            + New conversation
          </button>
          {conversations.length === 0 && (
            <p style={styles.emptyConv}>No conversations yet</p>
          )}
          {conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => handleSelectConversation(conv.id)}
              onMouseEnter={() => setHoveredConvId(conv.id)}
              onMouseLeave={() => setHoveredConvId(null)}
              style={{
                ...styles.convItem,
                color: theme.text,
                borderRadius: theme.borderRadius,
                backgroundColor: conv.id === activeConvId
                  ? theme.surfaceBg
                  : hoveredConvId === conv.id
                    ? `${theme.surfaceBg}88`
                    : "transparent",
              }}
            >
              <span style={styles.convTitle}>
                {conv.title || "Untitled"}
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteConversation(conv.id);
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.color = "#f87171";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.color = theme.textMuted;
                }}
                style={{
                  ...styles.deleteBtn,
                  color: theme.textMuted,
                  opacity: hoveredConvId === conv.id ? 1 : 0,
                }}
                title="Delete conversation"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Main chat area */}
      <main style={{
        ...styles.main,
        backgroundColor: theme.mainBg,
        transition: "background-color 0.4s ease, border-color 0.4s ease",
      }}>
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          style={{
            ...styles.toggleBtn,
            backgroundColor: theme.surfaceBg,
            border: `1px solid ${theme.border}`,
            color: theme.textMuted,
            borderRadius: theme.borderRadius,
          }}
          title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
        >
          {sidebarOpen ? "\u2039" : "\u203A"}
        </button>
        <ChatWindow
          precision={precision}
          onPrecisionChange={setPrecision}
          theme={theme}
          conversationId={activeConvId}
          messages={messages}
          onConversationCreated={handleConversationCreated}
          onMessagesChanged={handleMessagesChanged}
        />
      </main>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  layout: {
    display: "flex",
    height: "100vh",
  },
  sidebar: {
    minWidth: 0,
    display: "flex",
    flexDirection: "column",
    flexShrink: 0,
  },
  logo: {
    padding: "20px 16px",
    whiteSpace: "nowrap",
  },
  logoText: {
    margin: 0,
    fontSize: "22px",
    fontWeight: 700,
  },
  logoSubtext: {
    margin: "4px 0 0 0",
    fontSize: "11px",
  },
  convList: {
    flex: 1,
    overflowY: "auto",
    padding: "12px",
  },
  newConvBtn: {
    width: "100%",
    padding: "10px 12px",
    cursor: "pointer",
    fontSize: "13px",
    fontWeight: 500,
    textAlign: "left" as const,
    marginBottom: "12px",
  },
  emptyConv: {
    color: "#555",
    fontSize: "12px",
    textAlign: "center" as const,
    marginTop: "24px",
  },
  convItem: {
    padding: "8px 12px",
    fontSize: "13px",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: "4px",
    marginBottom: "2px",
  },
  convTitle: {
    flex: 1,
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  deleteBtn: {
    background: "none",
    border: "none",
    cursor: "pointer",
    padding: "2px",
    display: "flex",
    alignItems: "center",
    flexShrink: 0,
    transition: "opacity 0.15s ease, color 0.15s ease",
  },
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    position: "relative",
  },
  toggleBtn: {
    position: "absolute",
    top: "12px",
    left: "12px",
    zIndex: 10,
    width: "28px",
    height: "28px",
    cursor: "pointer",
    fontSize: "16px",
    lineHeight: "1",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
};
