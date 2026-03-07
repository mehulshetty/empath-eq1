/**
 * API client for communicating with the CLSA backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export interface PrecisionSettings {
  logic: number;
  eq: number;
}

export interface DeliberationMetadata {
  steps: number;
  module_precisions: Record<string, number>;
  converged: boolean;
  final_entropy: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  deliberation?: DeliberationMetadata | null;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  message_count: number;
}

export interface ConversationDetail {
  id: string;
  title: string;
  messages: ChatMessage[];
  created_at: string;
}

export async function createConversation(): Promise<ConversationSummary> {
  const res = await fetch(`${API_BASE}/conversations`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to create conversation");
  return res.json();
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const res = await fetch(`${API_BASE}/conversations`);
  if (!res.ok) throw new Error("Failed to list conversations");
  return res.json();
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  const res = await fetch(`${API_BASE}/conversations/${id}`);
  if (!res.ok) throw new Error("Failed to get conversation");
  return res.json();
}

export async function sendMessage(
  conversationId: string,
  content: string,
  precision: PrecisionSettings,
  returnDeliberation: boolean = true,
): Promise<{ message: ChatMessage; conversation_id: string }> {
  const res = await fetch(
    `${API_BASE}/conversations/${conversationId}/messages`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content,
        precision,
        return_deliberation: returnDeliberation,
      }),
    },
  );
  if (!res.ok) throw new Error("Failed to send message");
  return res.json();
}
