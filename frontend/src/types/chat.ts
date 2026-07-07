/** Role of a chat participant. */
export type MessageRole = "system" | "user" | "assistant";

/** A single message in the conversation. */
export interface Message {
  /** Client-side unique ID for React keys / optimistic updates. */
  id: string;
  role: MessageRole;
  content: string;
  /** ISO timestamp when the message was created. */
  createdAt: string;
}

/** The complete state exposed by useChat. */
export interface ChatState {
  messages: Message[];
  isStreaming: boolean;
  error: string | null;
  sendMessage: (content: string) => Promise<void>;
  clearMessages: () => void;
  clearError: () => void;
}

/** Shape of each SSE data payload from /api/chat/stream. */
export interface StreamChunk {
  content: string;
  done: boolean;
}
