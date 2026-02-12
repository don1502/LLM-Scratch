import { useState, useRef, useEffect } from 'react';

interface MessageInputProps {
    onSendMessage: (message: string) => void;
    isLoading: boolean;
}

export default function MessageInput({ onSendMessage, isLoading }: MessageInputProps) {
    const [input, setInput] = useState('');
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // Auto-resize textarea
    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
        }
    }, [input]);

    const handleSubmit = () => {
        const trimmed = input.trim();
        if (trimmed && !isLoading) {
            onSendMessage(trimmed);
            setInput('');
            if (textareaRef.current) {
                textareaRef.current.style.height = 'auto';
            }
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div className="border-t border-iron-border bg-iron-bg/80 backdrop-blur-sm px-6 md:px-10 py-4">
            <div className="w-full">
                <div
                    className={`
            relative flex items-end gap-2
            bg-iron-surface-light rounded-2xl
            border transition-all duration-200
            ${input ? 'border-arc-blue/40 shadow-[0_0_15px_rgba(79,195,247,0.1)]' : 'border-iron-border hover:border-iron-border-light'}
          `}
                >
                    {/* Textarea */}
                    <textarea
                        ref={textareaRef}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Message J.A.R.V.I.S..."
                        rows={1}
                        disabled={isLoading}
                        className="flex-1 bg-transparent text-iron-text text-sm
                       placeholder:text-iron-text-muted
                       resize-none py-3.5 pl-4 pr-2
                       max-h-[200px] leading-relaxed
                       disabled:opacity-50"
                    />

                    {/* Send Button */}
                    <div className="pr-2 pb-2">
                        <button
                            onClick={handleSubmit}
                            disabled={!input.trim() || isLoading}
                            className={`
                p-2 rounded-xl transition-all duration-200
                ${input.trim() && !isLoading
                                    ? 'bg-iron-red hover:bg-iron-red-light text-white shadow-[0_0_10px_rgba(192,57,43,0.3)] hover:shadow-[0_0_15px_rgba(192,57,43,0.5)]'
                                    : 'bg-iron-surface-hover text-iron-text-muted cursor-not-allowed'
                                }
              `}
                        >
                            {isLoading ? (
                                <div className="flex items-center gap-1 px-1">
                                    <div className="w-1.5 h-1.5 rounded-full bg-current animate-typing" style={{ animationDelay: '0ms' }} />
                                    <div className="w-1.5 h-1.5 rounded-full bg-current animate-typing" style={{ animationDelay: '200ms' }} />
                                    <div className="w-1.5 h-1.5 rounded-full bg-current animate-typing" style={{ animationDelay: '400ms' }} />
                                </div>
                            ) : (
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
                                </svg>
                            )}
                        </button>
                    </div>
                </div>


            </div>
        </div>
    );
}
