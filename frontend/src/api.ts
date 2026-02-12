/**
 * API service for communicating with the JARVIS GPT-2 backend.
 * All API calls to the Flask server go through here.
 */

const API_BASE_URL = '/api';

export interface ChatRequest {
    message: string;
    max_tokens?: number;
    temperature?: number;
    top_k?: number;
}

export interface ChatResponse {
    response: string;
    time_seconds?: number;
}

export interface HealthResponse {
    status: 'ok' | 'error';
    model: string | null;
    message: string;
}

export interface ApiError {
    error: string;
    message: string;
}

/**
 * Send a message to the GPT-2 model and get a generated response.
 */
export async function sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: request.message,
            max_tokens: request.max_tokens ?? 100,
            temperature: request.temperature ?? 0.7,
            top_k: request.top_k ?? 50,
        }),
    });

    if (!res.ok) {
        const errorData: ApiError = await res.json().catch(() => ({
            error: 'Network error',
            message: `Server returned status ${res.status}`,
        }));
        throw new Error(errorData.message || errorData.error || 'Unknown error');
    }

    return res.json();
}

/**
 * Check if the backend server and model are healthy.
 */
export async function checkHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
    });

    if (!res.ok) {
        const errorData = await res.json().catch(() => ({
            status: 'error' as const,
            model: null,
            message: `Server returned status ${res.status}`,
        }));
        return errorData;
    }

    return res.json();
}
