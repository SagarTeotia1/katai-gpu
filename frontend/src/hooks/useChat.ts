import { useCallback, useState } from "react";
import type { ChatState, Message, StreamChunk } from "@/types/chat";

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function now(): string {
  return new Date().toISOString();
}

export function useChat(): ChatState {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(async (content: string): Promise<void> => {
    const trimmed = content.trim();
    if (!trimmed || isStreaming) return;

    setError(null);

    // Add the user message immediately
    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content: trimmed,
      createdAt: now(),
    };

    // Build the full message list to send (including the new user message)
    const outgoingMessages = [...messages, userMessage];

    setMessages((prev) => [...prev, userMessage]);
    setIsStreaming(true);

    // Create an empty assistant message that we'll stream into
    const assistantId = generateId();
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      createdAt: now(),
    };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: outgoingMessages.map(({ role, content: c }) => ({
            role,
            content: c,
          })),
          stream: true,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Server error ${response.status}: ${text}`);
      }

      if (!response.body) {
        throw new Error("Response body is null — SSE not supported?");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by double newlines
        const parts = buffer.split("\n\n");
        // Keep the last (potentially incomplete) chunk in the buffer
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          // Each part may have multiple lines; find the "data:" line
          for (const line of part.split("\n")) {
            if (!line.startsWith("data: ")) continue;

            const raw = line.slice(6).trim(); // strip "data: "
            if (!raw) continue;

            let chunk: StreamChunk;
            try {
              chunk = JSON.parse(raw) as StreamChunk;
            } catch {
              continue;
            }

            if (chunk.done) {
              // Final sentinel — nothing more to append
              break;
            }

            if (chunk.content) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, content: msg.content + chunk.content }
                    : msg
                )
              );
            }
          }
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      // Remove the empty assistant placeholder on error
      setMessages((prev) =>
        prev.filter((msg) => !(msg.id === assistantId && msg.content === ""))
      );
    } finally {
      setIsStreaming(false);
    }
  }, [messages, isStreaming]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return { messages, isStreaming, error, sendMessage, clearMessages, clearError };
}
