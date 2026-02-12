export interface Message {
    id: string;
    role: 'user' | 'assistant' | 'error';
    content: string;
    timestamp: Date;
}

export interface Conversation {
    id: string;
    title: string;
    messages: Message[];
    createdAt: Date;
    updatedAt: Date;
}
