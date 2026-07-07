import {
  type FormEvent,
  type KeyboardEvent,
  useCallback,
  useRef,
  useState,
} from "react";
import { StreamingIndicator } from "./StreamingIndicator";

interface ChatInputProps {
  onSend: (content: string) => void;
  isStreaming: boolean;
}

export function ChatInput({ onSend, isStreaming }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = useCallback(
    (e?: FormEvent) => {
      e?.preventDefault();
      const trimmed = value.trim();
      if (!trimmed || isStreaming) return;
      onSend(trimmed);
      setValue("");
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    },
    [value, isStreaming, onSend]
  );

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      // Enter submits; Shift+Enter inserts newline
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  const handleInput = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setValue(e.target.value);
      // Auto-grow the textarea
      const ta = e.target;
      ta.style.height = "auto";
      ta.style.height = `${Math.min(ta.scrollHeight, 240)}px`;
    },
    []
  );

  const canSend = value.trim().length > 0 && !isStreaming;

  return (
    <div className="border-t border-gray-800 bg-gray-950 px-4 py-3">
      {/* Streaming status bar */}
      {isStreaming && (
        <div className="flex items-center gap-2 mb-2 text-xs text-gray-400">
          <StreamingIndicator />
          <span>Generating response…</span>
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        className="flex items-end gap-3 bg-gray-900 border border-gray-700 rounded-2xl px-4 py-2 focus-within:border-blue-500 transition-colors"
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          rows={1}
          placeholder={isStreaming ? "Waiting for response…" : "Message Qwen3…"}
          disabled={isStreaming}
          className="flex-1 resize-none bg-transparent text-gray-100 placeholder-gray-500 text-sm leading-relaxed focus:outline-none min-h-[1.5rem] max-h-60 disabled:opacity-60 disabled:cursor-not-allowed"
          aria-label="Chat message input"
          style={{ height: "auto" }}
        />

        <button
          type="submit"
          disabled={!canSend}
          aria-label="Send message"
          className={`flex-shrink-0 flex items-center justify-center w-9 h-9 rounded-xl transition-all duration-150 ${
            canSend
              ? "bg-blue-600 hover:bg-blue-500 text-white shadow-sm cursor-pointer"
              : "bg-gray-700 text-gray-500 cursor-not-allowed"
          }`}
        >
          {/* Send icon (paper plane) */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="w-4 h-4"
          >
            <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
          </svg>
        </button>
      </form>

      <p className="text-center text-xs text-gray-600 mt-2 select-none">
        Enter to send · Shift+Enter for newline
      </p>
    </div>
  );
}
