"""
Simple web UI to test pretrained and finetuned nanochat models.

Usage:
    python -m workshop.04_eval_finetune.scripts.chat_ui

Then open http://localhost:8080 in your browser.
"""

import argparse
import json
import os
import torch
import random
from contextlib import nullcontext
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator

from nanochat.common import compute_init, autodetect_device_type
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.engine import Engine


def get_workshop_dir():
    """Get the workshop/04_eval_finetune directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_models_dir():
    """Get the models directory for this workshop."""
    return os.path.join(get_workshop_dir(), "models")


def load_model_local(model_type: str, device, phase="eval"):
    """
    Load model from local workshop directory.

    model_type: "pretrained" or "finetuned/<tag>"
    """
    import re
    import glob as glob_module

    models_dir = get_models_dir()

    if model_type == "pretrained":
        checkpoints_dir = os.path.join(models_dir, "pretrained", "chatsft_checkpoints")
        tokenizer_dir = os.path.join(models_dir, "pretrained", "tokenizer")
    elif model_type.startswith("finetuned/"):
        tag = model_type.split("/", 1)[1]
        checkpoints_dir = os.path.join(models_dir, "finetuned", tag)
        tokenizer_dir = os.path.join(models_dir, "pretrained", "tokenizer")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Find model tag (for pretrained, it's d32)
    if model_type == "pretrained":
        model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
        if not model_tags:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
        # Pick largest model
        candidates = []
        for tag in model_tags:
            match = re.match(r"d(\d+)", tag)
            if match:
                candidates.append((int(match.group(1)), tag))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            model_tag = candidates[0][1]
        else:
            model_tag = model_tags[0]
        checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    else:
        # For finetuned models, checkpoints_dir is already the checkpoint dir
        checkpoint_dir = checkpoints_dir

    # Find latest step
    checkpoint_files = glob_module.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))

    print(f"Loading model from {checkpoint_dir} step {step}...")

    # Load checkpoint
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)

    # Convert bfloat16 to float for CPU/MPS
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }

    # Fix torch compile prefix
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build model
    model_config_kwargs = meta_data["model_config"]
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # Load tokenizer
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

    return model, tokenizer


# Parse arguments
parser = argparse.ArgumentParser(description='NanoChat Test UI')
parser.add_argument('-p', '--port', type=int, default=8080, help='Port to run the server on')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
parser.add_argument('--device-type', type=str, default='', help='Device type: cuda|cpu|mps (empty=autodetect)')
args = parser.parse_args()

# Initialize device
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Global model cache
loaded_models = {}


def get_model(model_type: str):
    """Get or load a model."""
    if model_type not in loaded_models:
        print(f"Loading {model_type}...")
        model, tokenizer = load_model_local(model_type, device)
        engine = Engine(model, tokenizer)
        loaded_models[model_type] = (model, tokenizer, engine)
    return loaded_models[model_type]


def list_available_models():
    """List all available models."""
    models = []
    models_dir = get_models_dir()

    # Check pretrained
    pretrained_path = os.path.join(models_dir, "pretrained", "chatsft_checkpoints")
    if os.path.exists(pretrained_path):
        models.append({"id": "pretrained", "name": "Pretrained (nanochat-d32)"})

    # Check finetuned
    finetuned_dir = os.path.join(models_dir, "finetuned")
    if os.path.exists(finetuned_dir):
        for tag in os.listdir(finetuned_dir):
            tag_path = os.path.join(finetuned_dir, tag)
            if os.path.isdir(tag_path):
                models.append({"id": f"finetuned/{tag}", "name": f"Finetuned ({tag})"})

    return models


# FastAPI app
app = FastAPI(title="NanoChat Test UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256
    top_k: Optional[int] = 50


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NanoChat Test UI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #fff;
            color: #333;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 100;
            background: #fff;
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            border-bottom: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .header h1 { font-size: 1.5rem; color: #e94560; flex-shrink: 0; }
        .header select {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: 1px solid #ddd;
            background: #fff;
            color: #333;
            font-size: 1rem;
        }
        .header .spacer { flex: 1; }
        .container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
            padding: 1rem;
            padding-top: 80px;
        }
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .message {
            padding: 1rem;
            border-radius: 12px;
            max-width: 80%;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        .message.user {
            background: #e94560;
            color: #fff;
            align-self: flex-end;
        }
        .message.assistant {
            background: #fff;
            align-self: flex-start;
            border: 1px solid #ddd;
        }
        .message.system {
            background: #e8e8e0;
            align-self: center;
            font-style: italic;
            color: #666;
        }
        .input-area {
            display: flex;
            gap: 0.5rem;
            padding: 1rem;
            background: #fff;
            border-radius: 12px;
            border: 1px solid #ddd;
        }
        .input-area textarea {
            flex: 1;
            padding: 0.75rem;
            border-radius: 8px;
            border: 1px solid #ddd;
            background: #fff;
            color: #333;
            font-size: 1rem;
            resize: none;
            min-height: 50px;
        }
        .input-area button {
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            border: none;
            background: #e94560;
            color: white;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .input-area button:hover { background: #ff6b6b; }
        .input-area button:disabled { background: #555; cursor: not-allowed; }
        .controls {
            display: flex;
            gap: 1rem;
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
            color: #666;
        }
        .controls label { display: flex; align-items: center; gap: 0.5rem; }
        .controls input {
            width: 60px;
            padding: 0.25rem;
            border-radius: 4px;
            border: 1px solid #ddd;
            background: #fff;
            color: #333;
        }
        .clear-btn {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: 1px solid #ddd;
            background: transparent;
            color: #666;
            cursor: pointer;
        }
        .clear-btn:hover { background: #eee; color: #333; }
        .loading { opacity: 0.7; }
    </style>
</head>
<body>
    <div class="header">
        <h1>NanoChat Test UI</h1>
        <select id="model-select">
            <option value="">Loading models...</option>
        </select>
        <div class="spacer"></div>
        <button class="clear-btn" onclick="clearChat()">Clear Chat</button>
    </div>

    <div class="container">
        <div class="chat-area" id="chat-area"></div>

        <div class="controls">
            <label>Temperature: <input type="number" id="temperature" value="0.7" step="0.1" min="0" max="2"></label>
            <label>Max tokens: <input type="number" id="max-tokens" value="256" step="32" min="1" max="1024"></label>
            <label>Top-k: <input type="number" id="top-k" value="50" step="10" min="1" max="200"></label>
        </div>

        <div class="input-area">
            <textarea id="user-input" placeholder="Type your message..." rows="2"></textarea>
            <button id="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let messages = [];

        async function loadModels() {
            const response = await fetch('/models');
            const models = await response.json();
            const select = document.getElementById('model-select');
            select.innerHTML = '';
            if (models.length === 0) {
                select.innerHTML = '<option value="">No models found</option>';
                return;
            }
            models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.id;
                opt.textContent = m.name;
                select.appendChild(opt);
            });
        }

        function addMessage(role, content) {
            messages.push({role, content});
            renderMessages();
        }

        function renderMessages() {
            const area = document.getElementById('chat-area');
            area.innerHTML = '';
            messages.forEach(m => {
                const div = document.createElement('div');
                div.className = 'message ' + m.role;
                div.textContent = m.content;
                area.appendChild(div);
            });
            area.scrollTop = area.scrollHeight;
        }

        function clearChat() {
            messages = [];
            renderMessages();
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const btn = document.getElementById('send-btn');
            const model = document.getElementById('model-select').value;
            const userText = input.value.trim();

            if (!userText || !model) return;

            addMessage('user', userText);
            input.value = '';
            btn.disabled = true;

            // Add placeholder for assistant
            messages.push({role: 'assistant', content: ''});
            renderMessages();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        model: model,
                        messages: messages.slice(0, -1),
                        temperature: parseFloat(document.getElementById('temperature').value),
                        max_tokens: parseInt(document.getElementById('max-tokens').value),
                        top_k: parseInt(document.getElementById('top-k').value)
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantText = '';

                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.token) {
                                    assistantText += data.token;
                                    messages[messages.length - 1].content = assistantText;
                                    renderMessages();
                                }
                            } catch (e) {}
                        }
                    }
                }
            } catch (error) {
                messages[messages.length - 1].content = 'Error: ' + error.message;
                renderMessages();
            }

            btn.disabled = false;
        }

        // Handle Enter key
        document.getElementById('user-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Load models on page load
        loadModels();
    </script>
</body>
</html>
"""


@app.get("/")
async def root():
    """Serve the chat UI."""
    return HTMLResponse(content=HTML_TEMPLATE)


@app.get("/models")
async def get_models():
    """List available models."""
    return list_available_models()


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat completion endpoint with streaming."""
    try:
        model, tokenizer, engine = get_model(request.model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Build conversation tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    conversation_tokens = [bos]
    for message in request.messages:
        if message.role == "user":
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(message.content))
            conversation_tokens.append(user_end)
        elif message.role == "assistant":
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(tokenizer.encode(message.content))
            conversation_tokens.append(assistant_end)

    conversation_tokens.append(assistant_start)

    async def generate():
        accumulated_tokens = []
        last_clean_text = ""

        with autocast_ctx:
            for token_column, token_masks in engine.generate(
                conversation_tokens,
                num_samples=1,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                seed=random.randint(0, 2**31 - 1)
            ):
                token = token_column[0]

                if token == assistant_end or token == bos:
                    break

                accumulated_tokens.append(token)
                current_text = tokenizer.decode(accumulated_tokens)

                if not current_text.endswith('�'):
                    new_text = current_text[len(last_clean_text):]
                    if new_text:
                        yield f"data: {json.dumps({'token': new_text}, ensure_ascii=False)}\n\n"
                        last_clean_text = current_text

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Test UI on http://localhost:{args.port}")
    print(f"Models directory: {get_models_dir()}")
    print(f"Available models: {list_available_models()}")
    uvicorn.run(app, host=args.host, port=args.port)
