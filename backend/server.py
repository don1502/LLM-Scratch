"""
Flask API server for the GPT-2 JARVIS chatbot.
Exposes /api/chat and /api/health endpoints.
"""

import sys
import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate import load_model, generate

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend dev server

# Global model state
model = None
tokenizer = None
model_loaded = False
model_error = None


def init_model():
    """Load GPT-2 model on startup."""
    global model, tokenizer, model_loaded, model_error
    try:
        print("=" * 50)
        print("Loading GPT-2 124M model...")
        print("This may take 10-30 seconds on first load...")
        print("=" * 50)
        start = time.time()
        model, tokenizer = load_model(
            models_dir=os.path.join(os.path.dirname(__file__), "gpt_models"),
            model_size="124M"
        )
        elapsed = time.time() - start
        model_loaded = True
        print(f"Model loaded in {elapsed:.1f}s")
        print("=" * 50)
    except Exception as e:
        model_error = str(e)
        model_loaded = False
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    if model_loaded:
        return jsonify({
            "status": "ok",
            "model": "GPT-2 124M",
            "message": "Model loaded and ready for inference"
        })
    else:
        return jsonify({
            "status": "error",
            "model": None,
            "message": model_error or "Model not loaded yet"
        }), 503


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat endpoint — generates text from a prompt.
    
    Request body:
        {
            "message": "Your prompt text",
            "max_tokens": 100,       (optional, default 100)
            "temperature": 0.7,      (optional, default 0.7)
            "top_k": 50              (optional, default 50)
        }
    
    Response:
        {
            "response": "Generated text...",
            "tokens_generated": 42
        }
    """
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded",
            "message": model_error or "The model is still loading. Please try again."
        }), 503

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({
            "error": "Bad request",
            "message": "Request body must contain a 'message' field"
        }), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({
            "error": "Bad request",
            "message": "Message cannot be empty"
        }), 400

    max_tokens = min(data.get("max_tokens", 100), 500)  # Cap at 500
    temperature = max(0.0, min(data.get("temperature", 0.7), 2.0))  # Clamp 0-2
    top_k = max(0, min(data.get("top_k", 50), 100))  # Clamp 0-100

    try:
        start = time.time()

        # Determine device
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=user_message,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )

        elapsed = time.time() - start

        return jsonify({
            "response": generated_text.strip(),
            "time_seconds": round(elapsed, 2),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Generation failed",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    init_model()
    print("\nStarting JARVIS API server on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /api/health  - Health check")
    print("  POST /api/chat    - Generate text")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False)
