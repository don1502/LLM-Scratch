import { type Message } from '../types';

interface MessageBubbleProps {
    message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
    const isUser = message.role === 'user';
    const isError = message.role === 'error';

    const copyToClipboard = () => {
        navigator.clipboard.writeText(message.content);
    };

    return (
        <div className={`animate-slide-up ${isUser ? 'flex justify-end' : ''}`}>
            <div className={`flex gap-3 max-w-[720px] ${isUser ? 'flex-row-reverse' : ''} mx-auto w-full px-4`}>
                {/* Avatar */}
                <div className="flex-shrink-0 mt-1">
                    {isError ? (
                        <div className="w-8 h-8 rounded-full bg-red-500/20 border border-red-500/40 flex items-center justify-center">
                            <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                            </svg>
                        </div>
                    ) : isUser ? (
                        <div className="w-8 h-8 rounded-full bg-iron-gold/20 border border-iron-gold/40 flex items-center justify-center">
                            <svg className="w-4 h-4 text-iron-gold" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                        </div>
                    ) : (
                        <div className="relative w-8 h-8">
                            <div className="absolute inset-0 rounded-full bg-arc-blue/15" />
                            <div className="absolute inset-0.5 rounded-full border border-arc-blue/50" />
                            <div className="absolute inset-2 rounded-full bg-arc-blue/30" />
                            <div className="absolute inset-2.5 rounded-full bg-arc-blue shadow-[0_0_8px_rgba(79,195,247,0.5)]" />
                        </div>
                    )}
                </div>

                {/* Message Content */}
                <div className={`flex-1 min-w-0 ${isUser ? 'text-right' : ''}`}>
                    {/* Role label */}
                    <div className={`text-xs font-medium mb-1.5 ${isError ? 'text-red-400/70' : isUser ? 'text-iron-gold/70' : 'text-arc-blue/70'}`}>
                        {isError ? 'Error' : isUser ? 'You' : 'J.A.R.V.I.S.'}
                    </div>

                    {/* Message bubble */}
                    <div
                        className={`
              inline-block text-left rounded-2xl px-4 py-3 text-sm leading-relaxed
              ${isError
                                ? 'bg-red-500/10 border border-red-500/30 text-red-300'
                                : isUser
                                    ? 'bg-iron-gold/10 border border-iron-gold/20 text-iron-text'
                                    : 'bg-iron-surface-light border border-iron-border text-iron-text'
                            }
            `}
                    >
                        <div className="whitespace-pre-wrap break-words">{message.content}</div>
                    </div>

                    {/* Action buttons */}
                    {!isUser && !isError && (
                        <div className="flex items-center gap-2 mt-2 opacity-0 group-hover:opacity-100 hover:opacity-100 transition-opacity">
                            <button
                                onClick={copyToClipboard}
                                className="flex items-center gap-1 px-2 py-1 rounded-md text-xs text-iron-text-muted
                           hover:text-iron-text hover:bg-iron-surface-hover transition-colors"
                                title="Copy message"
                            >
                                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                        d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                </svg>
                                Copy
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
