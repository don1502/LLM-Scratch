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
        <div className="flex-shrink-0 w-full px-6 md:px-10 lg:px-16 pb-5 pt-3">
            <div
                className={`
                    relative flex items-end
                    bg-iron-surface-light rounded-2xl
                    border transition-all duration-300
                    ${input
                        ? 'border-arc-blue/50 shadow-[0_0_20px_rgba(79,195,247,0.12),0_4px_16px_rgba(0,0,0,0.2)]'
                        : 'border-iron-border hover:border-iron-border-light shadow-[0_2px_12px_rgba(0,0,0,0.15)]'
                    }
                `}
            >
                {/* Textarea */}
                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Message LLM..."
                    rows={1}
                    disabled={isLoading}
                    className="flex-1 bg-transparent text-iron-text text-sm
                        placeholder:text-iron-text-muted/60 placeholder:text-sm
                        resize-none py-4 pl-5 pr-3
                        min-h-[56px] max-h-[200px] leading-relaxed
                        focus:outline-none
                        disabled:opacity-50"
                    style={{ verticalAlign: 'middle' }}
                />

                {/* Send Button */}
                <div className="pr-3 pb-3 flex-shrink-0">
                    <button
                        onClick={handleSubmit}
                        disabled={!input.trim() || isLoading}
                        className={`
                            p-2.5 rounded-xl transition-all duration-200
                            ${input.trim() && !isLoading
                                ? 'bg-arc-blue hover:bg-arc-blue/80 text-white shadow-[0_0_12px_rgba(79,195,247,0.4)] hover:shadow-[0_0_18px_rgba(79,195,247,0.6)] hover:scale-105'
                                : 'bg-iron-surface-hover text-iron-text-muted cursor-not-allowed'
                            }
                        `}
                    >
                        {isLoading ? (
                            <div className="flex items-center gap-1 px-0.5">
                                <div className="w-1.5 h-1.5 rounded-full bg-current animate-typing" style={{ animationDelay: '0ms' }} />
                                <div className="w-1.5 h-1.5 rounded-full bg-current animate-typing" style={{ animationDelay: '200ms' }} />
                                <div className="w-1.5 h-1.5 rounded-full bg-current animate-typing" style={{ animationDelay: '400ms' }} />
                            </div>
                        ) : (
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4.5 10.5L12 3m0 0l7.5 7.5M12 3v18" />
                            </svg>
                        )}
                    </button>
                </div>
            </div>

            {/* Subtle hint */}
            <p className="text-center text-[11px] text-iron-text-muted/40 mt-2">
                Press Enter to send · Shift + Enter for new line
            </p>
        </div>
    );
}
