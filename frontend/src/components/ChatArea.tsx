import { useRef, useEffect } from 'react';
import { type Message } from '../types';
import MessageBubble from './MessageBubble';
import WelcomeScreen from './WelcomeScreen';
import MessageInput from './MessageInput';

interface ChatAreaProps {
    messages: Message[];
    onSendMessage: (message: string) => void;
    isLoading: boolean;
    isConnected: boolean;
    onToggleSidebar: () => void;
}

export default function ChatArea({ messages, onSendMessage, isLoading, isConnected, onToggleSidebar }: ChatAreaProps) {
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);



    return (
        <div className="flex-1 flex flex-col h-full min-w-0">
            {/* Top Bar */}
            <header className="flex items-center h-14 px-4 border-b border-iron-border bg-iron-surface/50 backdrop-blur-sm flex-shrink-0">
                {/* Mobile hamburger */}
                <button
                    onClick={onToggleSidebar}
                    className="lg:hidden p-2 -ml-2 mr-2 rounded-lg hover:bg-iron-surface-hover text-iron-text-secondary hover:text-iron-text transition-colors"
                >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                </button>

                {/* Model selector */}
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-arc-blue shadow-[0_0_6px_rgba(79,195,247,0.8)]" />
                    <span className="font-display text-sm font-medium text-iron-text tracking-wide">
                        J.A.R.V.I.S.
                    </span>
                    <span className="text-xs text-iron-text-muted px-2 py-0.5 rounded-md bg-iron-surface-light border border-iron-border">
                        GPT — From Scratch
                    </span>
                </div>

                {/* Right side actions */}
                <div className="ml-auto flex items-center gap-1">
                    <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-lg bg-iron-surface-light border border-iron-border">
                        <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-green-400 shadow-[0_0_4px_rgba(74,222,128,0.6)]' : 'bg-red-400 shadow-[0_0_4px_rgba(248,113,113,0.6)]'}`} />
                        <span className="text-xs text-iron-text-muted">{isConnected ? 'Online' : 'Offline'}</span>
                    </div>
                </div>
            </header>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto">
                {messages.length === 0 ? (
                    <WelcomeScreen />
                ) : (
                    <div className="py-6 space-y-6">
                        {messages.map((message) => (
                            <MessageBubble key={message.id} message={message} />
                        ))}

                        {/* Typing indicator */}
                        {isLoading && (
                            <div className="flex gap-3 max-w-[720px] mx-auto px-4 animate-fade-in">
                                <div className="relative w-8 h-8 flex-shrink-0 mt-1">
                                    <div className="absolute inset-0 rounded-full bg-arc-blue/15" />
                                    <div className="absolute inset-0.5 rounded-full border border-arc-blue/50" />
                                    <div className="absolute inset-2 rounded-full bg-arc-blue/30" />
                                    <div className="absolute inset-2.5 rounded-full bg-arc-blue shadow-[0_0_8px_rgba(79,195,247,0.5)]" />
                                </div>
                                <div className="bg-iron-surface-light border border-iron-border rounded-2xl px-4 py-3">
                                    <div className="flex items-center gap-1.5">
                                        <div className="w-2 h-2 rounded-full bg-arc-blue animate-typing" style={{ animationDelay: '0ms' }} />
                                        <div className="w-2 h-2 rounded-full bg-arc-blue animate-typing" style={{ animationDelay: '200ms' }} />
                                        <div className="w-2 h-2 rounded-full bg-arc-blue animate-typing" style={{ animationDelay: '400ms' }} />
                                    </div>
                                </div>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Message Input */}
            <MessageInput onSendMessage={onSendMessage} isLoading={isLoading} />
        </div>
    );
}
