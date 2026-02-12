import { useState, useCallback, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import { type Message, type Conversation } from './types';
import { sendMessage, checkHealth } from './api';

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  const activeConversation = conversations.find((c) => c.id === activeConversationId) ?? null;

  // Check backend health on mount and periodically
  useEffect(() => {
    const check = async () => {
      try {
        const health = await checkHealth();
        setIsConnected(health.status === 'ok');
      } catch {
        setIsConnected(false);
      }
    };

    check();
    const interval = setInterval(check, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  const createNewChat = useCallback(() => {
    const newConv: Conversation = {
      id: generateId(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    setConversations((prev) => [newConv, ...prev]);
    setActiveConversationId(newConv.id);
    setSidebarOpen(false);
  }, []);

  const handleSendMessage = useCallback(
    async (content: string) => {
      let currentConvId = activeConversationId;

      // If no active conversation, create one
      if (!currentConvId) {
        const newConv: Conversation = {
          id: generateId(),
          title: content.slice(0, 40) + (content.length > 40 ? '...' : ''),
          messages: [],
          createdAt: new Date(),
          updatedAt: new Date(),
        };
        setConversations((prev) => [newConv, ...prev]);
        setActiveConversationId(newConv.id);
        currentConvId = newConv.id;
      }

      const userMessage: Message = {
        id: generateId(),
        role: 'user',
        content,
        timestamp: new Date(),
      };

      // Add user message and update title if it's the first message
      setConversations((prev) =>
        prev.map((conv) => {
          if (conv.id === currentConvId) {
            const isFirstMessage = conv.messages.length === 0;
            return {
              ...conv,
              title: isFirstMessage ? content.slice(0, 40) + (content.length > 40 ? '...' : '') : conv.title,
              messages: [...conv.messages, userMessage],
              updatedAt: new Date(),
            };
          }
          return conv;
        })
      );

      // Call the real backend API
      setIsLoading(true);
      try {
        const data = await sendMessage({ message: content });

        const assistantMessage: Message = {
          id: generateId(),
          role: 'assistant',
          content: data.response,
          timestamp: new Date(),
        };

        setConversations((prev) =>
          prev.map((conv) => {
            if (conv.id === currentConvId) {
              return {
                ...conv,
                messages: [...conv.messages, assistantMessage],
                updatedAt: new Date(),
              };
            }
            return conv;
          })
        );
        setIsConnected(true);
      } catch (err) {
        // Show error message in chat
        const errorMessage: Message = {
          id: generateId(),
          role: 'error',
          content: err instanceof Error
            ? err.message
            : 'Failed to connect to JARVIS backend. Make sure the server is running on port 5000.',
          timestamp: new Date(),
        };

        setConversations((prev) =>
          prev.map((conv) => {
            if (conv.id === currentConvId) {
              return {
                ...conv,
                messages: [...conv.messages, errorMessage],
                updatedAt: new Date(),
              };
            }
            return conv;
          })
        );
        setIsConnected(false);
      } finally {
        setIsLoading(false);
      }
    },
    [activeConversationId]
  );

  return (
    <div className="flex h-full bg-iron-bg">
      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={(id) => {
          setActiveConversationId(id);
          setSidebarOpen(false);
        }}
        onNewChat={createNewChat}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />
      <ChatArea
        messages={activeConversation?.messages ?? []}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
        isConnected={isConnected}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
      />
    </div>
  );
}
