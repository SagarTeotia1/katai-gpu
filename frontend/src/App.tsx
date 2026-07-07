import { useEffect, useState } from "react";
import { ChatInput } from "@/components/ChatInput";
import { MessageList } from "@/components/MessageList";
import { useChat } from "@/hooks/useChat";

const DEFAULT_MODEL = "qwen3.6:27b-bf16";

export default function App() {
  const { messages, isStreaming, error, sendMessage, clearMessages, clearError } =
    useChat();
  const [modelId, setModelId] = useState<string>(DEFAULT_MODEL);

  // Fetch the active model name from vLLM via the backend
  useEffect(() => {
    let cancelled = false;
    fetch("/api/models")
      .then((r) => r.json())
      .then((data: { models?: string[] }) => {
        if (!cancelled && data.models && data.models.length > 0 && data.models[0]) {
          setModelId(data.models[0]);
        }
      })
      .catch(() => {
        // silently ignore — use default label
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Short display name (strip org prefix for display)
  const displayName = modelId.includes("/") ? (modelId.split("/")[1] ?? modelId) : modelId;

  return (
    <div className="dark flex flex-col h-full bg-gray-950">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-gray-800 bg-gray-950 z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-600 to-blue-500 flex items-center justify-center text-white text-sm font-bold shadow-sm">
            Q
          </div>
          <div>
            <h1 className="text-white font-semibold text-sm leading-none">{displayName}</h1>
            <p className="text-gray-500 text-xs mt-0.5">Local GPU inference · Ollama</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Status dot */}
          <span className="flex items-center gap-1.5 text-xs text-gray-400">
            <span
              className={`w-2 h-2 rounded-full ${
                isStreaming ? "bg-yellow-400 animate-pulse" : "bg-emerald-400"
              }`}
            />
            {isStreaming ? "Generating" : "Ready"}
          </span>

          {/* Clear chat button */}
          {messages.length > 0 && (
            <button
              onClick={clearMessages}
              disabled={isStreaming}
              className="ml-3 text-xs text-gray-500 hover:text-gray-300 transition-colors disabled:opacity-40 disabled:cursor-not-allowed px-2 py-1 rounded-lg hover:bg-gray-800"
              title="Clear conversation"
            >
              Clear
            </button>
          )}
        </div>
      </header>

      {/* ── Error banner ────────────────────────────────────────────────── */}
      {error && (
        <div className="mx-4 mt-3 flex items-start gap-3 bg-red-950 border border-red-800 rounded-xl px-4 py-3">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
            className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5"
          >
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zm0-11a1 1 0 10-2 0v4a1 1 0 102 0V7zm0 6a1 1 0 100-2 1 1 0 000 2z"
              clipRule="evenodd"
            />
          </svg>
          <div className="flex-1">
            <p className="text-red-300 text-sm font-medium">Error</p>
            <p className="text-red-400 text-xs mt-0.5 break-words">{error}</p>
          </div>
          <button
            onClick={clearError}
            className="text-red-500 hover:text-red-300 transition-colors"
            aria-label="Dismiss error"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="w-4 h-4"
            >
              <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
            </svg>
          </button>
        </div>
      )}

      {/* ── Message area ────────────────────────────────────────────────── */}
      <MessageList messages={messages} isStreaming={isStreaming} />

      {/* ── Input ───────────────────────────────────────────────────────── */}
      <ChatInput onSend={sendMessage} isStreaming={isStreaming} />
    </div>
  );
}
