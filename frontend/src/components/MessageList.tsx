import { useEffect, useRef } from "react";
import type { Message as MessageType } from "@/types/chat";
import { Message } from "./Message";

interface MessageListProps {
  messages: MessageType[];
  isStreaming: boolean;
}

export function MessageList({ messages, isStreaming }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom whenever messages change or streaming updates
  useEffect(() => {
    if (!bottomRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const { scrollTop, scrollHeight, clientHeight } = container;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 120;

    // Only auto-scroll if the user is already near the bottom
    if (isNearBottom || isStreaming) {
      bottomRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages, isStreaming]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center px-4">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-600 to-blue-500 flex items-center justify-center text-white text-3xl font-bold mb-4 shadow-lg">
          Q
        </div>
        <h2 className="text-xl font-semibold text-white mb-2">qwen3.6:27b-bf16</h2>
        <p className="text-gray-400 text-sm max-w-sm leading-relaxed">
          Local GPU inference via Ollama on A100 80GB. Responses stream in real-time.
          Ask anything to get started.
        </p>
        <div className="mt-6 flex flex-wrap gap-2 justify-center">
          {[
            "Explain quantum entanglement simply",
            "Write a Python async generator",
            "What is the difference between TCP and UDP?",
          ].map((suggestion) => (
            <button
              key={suggestion}
              className="text-xs text-gray-400 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-full px-3 py-1.5 transition-colors cursor-default"
              aria-hidden="true"
              tabIndex={-1}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto px-4 py-4 space-y-1"
    >
      {messages.map((message, index) => (
        <Message
          key={message.id}
          message={message}
          isLastMessage={index === messages.length - 1}
          isStreaming={isStreaming}
        />
      ))}
      {/* Invisible anchor for scroll-to-bottom */}
      <div ref={bottomRef} className="h-1" aria-hidden="true" />
    </div>
  );
}
