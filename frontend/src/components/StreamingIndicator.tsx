/**
 * Animated three-dot "thinking" indicator shown while the model streams.
 */
export function StreamingIndicator() {
  return (
    <div
      className="flex items-center gap-1.5 px-1 py-0.5"
      role="status"
      aria-label="Assistant is typing"
    >
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="inline-block w-2 h-2 rounded-full bg-blue-400 animate-bounce-dot"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </div>
  );
}
