import { type Conversation } from '../types';

interface SidebarProps {
    conversations: Conversation[];
    activeConversationId: string | null;
    onSelectConversation: (id: string) => void;
    onNewChat: () => void;
    isOpen: boolean;
    onToggle: () => void;
}

export default function Sidebar({
    conversations,
    activeConversationId,
    onSelectConversation,
    onNewChat,
    isOpen,
    onToggle,
}: SidebarProps) {
    return (
        <>
            {/* Mobile overlay */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-40 lg:hidden"
                    onClick={onToggle}
                />
            )}

            <aside
                className={`
          fixed lg:relative z-50 h-full
          w-[280px] flex flex-col
          bg-iron-surface border-r border-iron-border
          transition-transform duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          lg:translate-x-0
        `}
            >
                {/* Header / Brand */}
                <div className="p-4 border-b border-iron-border">
                    <button
                        onClick={onNewChat}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-xl
                       bg-iron-surface-light hover:bg-iron-surface-hover
                       border border-iron-border hover:border-arc-blue/40
                       transition-all duration-200 group"
                    >
                        {/* Arc Reactor Icon */}
                        <div className="relative w-8 h-8 flex-shrink-0">
                            <div className="absolute inset-0 rounded-full bg-arc-blue/20 animate-arc-pulse" />
                            <div className="absolute inset-1 rounded-full border-2 border-arc-blue/60" />
                            <div className="absolute inset-2.5 rounded-full bg-arc-blue/40" />
                            <div className="absolute inset-3 rounded-full bg-arc-blue shadow-[0_0_10px_rgba(79,195,247,0.6)]" />
                        </div>
                        <span className="font-display text-sm font-semibold text-iron-text group-hover:text-arc-blue transition-colors">
                            New Chat
                        </span>
                        <svg
                            className="w-4 h-4 ml-auto text-iron-text-muted group-hover:text-arc-blue transition-colors"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                    </button>
                </div>

                {/* Conversation List */}
                <div className="flex-1 overflow-y-auto py-2 px-2">
                    {conversations.length === 0 ? (
                        <div className="px-4 py-8 text-center">
                            <p className="text-iron-text-muted text-sm">No conversations yet</p>
                            <p className="text-iron-text-muted text-xs mt-1">Start a new chat to begin</p>
                        </div>
                    ) : (
                        <div className="space-y-1">
                            {conversations.map((conv) => (
                                <button
                                    key={conv.id}
                                    onClick={() => onSelectConversation(conv.id)}
                                    className={`
                    w-full text-left px-3 py-2.5 rounded-lg
                    transition-all duration-150 group
                    ${activeConversationId === conv.id
                                            ? 'bg-iron-surface-hover border border-iron-gold/30 text-iron-text'
                                            : 'hover:bg-iron-surface-hover border border-transparent text-iron-text-secondary hover:text-iron-text'
                                        }
                  `}
                                >
                                    <div className="flex items-center gap-2.5">
                                        <svg
                                            className={`w-4 h-4 flex-shrink-0 ${activeConversationId === conv.id ? 'text-iron-gold' : 'text-iron-text-muted'
                                                }`}
                                            fill="none"
                                            stroke="currentColor"
                                            viewBox="0 0 24 24"
                                        >
                                            <path
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                                strokeWidth={1.5}
                                                d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                                            />
                                        </svg>
                                        <span className="truncate text-sm">{conv.title}</span>
                                    </div>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            </aside>
        </>
    );
}
