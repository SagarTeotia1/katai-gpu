import ReactMarkdown from "react-markdown";
import type { Message as MessageType } from "@/types/chat";
import { StreamingIndicator } from "./StreamingIndicator";

interface MessageProps {
  message: MessageType;
  isLastMessage: boolean;
  isStreaming: boolean;
}

export function Message({ message, isLastMessage, isStreaming }: MessageProps) {
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  const showStreamingIndicator =
    isAssistant && isLastMessage && isStreaming && message.content === "";

  return (
    <div
      className={`flex w-full mb-4 ${isUser ? "justify-end" : "justify-start"}`}
    >
      {/* Avatar */}
      {isAssistant && (
        <div className="flex-shrink-0 mr-3 mt-0.5">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-600 to-blue-500 flex items-center justify-center text-white text-xs font-bold select-none">
            Q
          </div>
        </div>
      )}

      {/* Bubble */}
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-blue-600 text-white rounded-tr-sm"
            : "bg-gray-800 text-gray-100 rounded-tl-sm border border-gray-700/50"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap break-words leading-relaxed text-sm">
            {message.content}
          </p>
        ) : showStreamingIndicator ? (
          <StreamingIndicator />
        ) : (
          <div className="prose-dark text-sm">
            <ReactMarkdown
              components={{
                // Render inline code
                code({ className, children, ...props }) {
                  const isBlock = className?.startsWith("language-");
                  if (isBlock) {
                    return (
                      <pre className="bg-gray-900 rounded-lg p-4 overflow-x-auto my-3 border border-gray-700">
                        <code className="text-sm text-gray-200 font-mono" {...props}>
                          {children}
                        </code>
                      </pre>
                    );
                  }
                  return (
                    <code
                      className="bg-gray-900 text-emerald-400 rounded px-1.5 py-0.5 text-[0.85em] font-mono"
                      {...props}
                    >
                      {children}
                    </code>
                  );
                },
                // Open links in a new tab
                a({ href, children, ...props }) {
                  return (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 underline hover:text-blue-300 transition-colors"
                      {...props}
                    >
                      {children}
                    </a>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {/* User avatar */}
      {isUser && (
        <div className="flex-shrink-0 ml-3 mt-0.5">
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold select-none">
            U
          </div>
        </div>
      )}
    </div>
  );
}
