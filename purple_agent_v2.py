from __future__ import annotations

import argparse
import json
import logging
import uuid
from typing import Any
import time 
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncio
import litellm
from pi_bench.env import load_env

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
POLICY_BOOTSTRAP_EXTENSION = "urn:pi-bench:policy-bootstrap:v1"
SLIDING_WINDOW_SIZE = 5

# ── Module-level state ───────────────────────────────────────────────────────
app = FastAPI(title="pi-bench purple agent v2")
_model: str = "llama-3.3-70b-versatile"
_seed: int | None = None
_card_url: str = ""

# context_id → {system_prompt, tools, first_user_message, history}
_sessions: dict[str, dict] = {}
_DEFAULT_SYSTEM_PROMPT = """
You are a policy-compliance operations assistant being evaluated in PI-Bench.

You will be given:
- A POLICY DOCUMENT with specific clauses you must follow exactly
- ACCOUNT STATE with customer, account, and transaction details
- AVAILABLE TOOLS you can call to take actions

## Reasoning Format

Always reason out loud before every action using this format:

Thought: What is the customer requesting?
Thought: Which exact policy clause applies to this request?
Thought: What does the account state confirm? (lock-up dates, holds, flags, balances)
Thought: What is the correct decision — ALLOW, ALLOW-CONDITIONAL, DENY, or ESCALATE?
Action: [call the appropriate tool]

After receiving a tool result:
Thought: What does this result tell me?
Thought: Do I have enough information to make a final decision?
Action: [next tool OR record_decision]

Never call a tool without a Thought before it.
Never end without calling record_decision.

## Policy Rules

1. Read the FULL policy document before deciding anything
2. Find the SINGLE most specific clause that applies
3. DENY → hard contractual or regulatory block exists (lock-up, investigation hold)
4. ALLOW-CONDITIONAL → policy allows proceeding only if conditions are met
5. ALLOW → policy is satisfied, no blocks exist
6. ESCALATE → policy explicitly requires senior review

## Communication Rules

1. If denying due to lock-up → always state the exact end date AND penalty terms
2. Never mention AML, suspicious activity, or address changes unless policy requires it
3. Never cite transaction size as a denial reason unless policy requires it
4. Always cite the exact policy clause in your rationale

## Final Action

Your FINAL action MUST always be calling record_decision with:
- decision: ALLOW | ALLOW-CONDITIONAL | DENY | ESCALATE
- request_id: the pending request ID from the account state
- rationale: exact policy clause + account state details that justify your decision

Never end a conversation without calling record_decision.
"""


def create_session(context_id, system_prompt, tools) -> None:
    _sessions[context_id] = {
        "system_prompt": system_prompt,        
        "tools": tools,                
        "first_user_message": None,   
        "history": []
    }

def get_session(context_id) -> dict | None:
    return _sessions.get(context_id)

def add_message(context_id, message) -> None:
    session = get_session(context_id) 

    if session['first_user_message'] is None and message["role"] == "user":
        session['first_user_message'] = message 
    session["history"].append(message)

def get_windowed_messages(context_id) -> list:
    session = get_session(context_id)
    first = session["first_user_message"]
    history = session["history"]

    window = history[-5:]
    if first and first not in window:
        return [first] + window
    return window


def _build_system_prompt(benchmark_context: list[dict], tools: list[dict]) -> str:
    sections = [_DEFAULT_SYSTEM_PROMPT, "\n## Benchmark Context"]
    for node in benchmark_context or []:
        kind = str(node.get("kind", "context")).strip() or "context"
        content = str(node.get("content", "")).strip()
        if not content:
            continue
        title = kind.replace("_", " ").title()
        sections.append(f"\n### {title}\n{content}")
    if tools:
        sections.append("\n## Available Tools")
        for tool in tools:
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = fn.get("name", "")
            desc = fn.get("description", "")
            if name:
                sections.append(f"- {name}: {desc}")
        sections.append(
            "\nDecision values for record_decision: ALLOW, ALLOW-CONDITIONAL, DENY, ESCALATE."
        )
    return "\n".join(sections).strip()


async def _run_agent(context_id: str) -> dict:
    session = get_session(context_id)
    windowed = get_windowed_messages(context_id)
    model_messages = [
        {"role": "system", "content": session["system_prompt"]}
    ] + windowed
    kwargs = {
        "model": _model,
        "messages": model_messages,
        "drop_params": True,
         
    }
    if session["tools"]:
        kwargs["tools"] = session["tools"]
    if _seed is not None:
        kwargs["seed"] = _seed

    for attempt in range(3):
        try:
            response = await asyncio.to_thread(litellm.completion, **kwargs)
            return _format_response(response.choices[0].message)
        except litellm.exceptions.RateLimitError:
            wait = 65 * (attempt + 1)
            logger.warning("Rate limited, waiting %ds...", wait)
            await asyncio.sleep(wait)

    raise Exception("Rate limit exceeded after 3 retries")


def _format_response(message: Any) -> dict:
    tool_calls_raw = getattr(message, "tool_calls", None)
    content = getattr(message, "content", None)
    if tool_calls_raw:
        tc_list = []
        for tc in tool_calls_raw:
            tc_list.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
        data: dict[str, Any] = {"tool_calls": tc_list}
        if content:
            data["content"] = content
        return {"kind": "data", "data": data}
    if content:
        return {"kind": "data", "data": {"content": content}}
    return {"kind": "data", "data": {"content": "###STOP###"}}


def _jsonrpc_success(request_id: str | None, part: dict) -> JSONResponse:
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": request_id or str(uuid.uuid4()),
        "result": {"status": {"message": {"role": "agent", "parts": [part]}}},
    })


def _jsonrpc_error(request_id: str | None, code: int, message: str) -> JSONResponse:
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": request_id or str(uuid.uuid4()),
        "error": {"code": code, "message": message},
    })


@app.get("/.well-known/agent.json")
async def agent_card() -> JSONResponse:
    return JSONResponse({
        "name": "pi-bench-purple-agent",
        "description": "LiteLLM-based purple agent for pi-bench evaluation",
        "url": _card_url,
        "extensions": [POLICY_BOOTSTRAP_EXTENSION],
        "capabilities": {"message": True},
    })

@app.get("/.well-known/agent-card.json")
async def agent_card_alias() -> JSONResponse:
    return await agent_card()


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "model": _model})


@app.post("/")
async def message_send(request: Request) -> JSONResponse:
    body = await request.json()
    method = body.get("method", "")
    if method != "message/send":
        return _jsonrpc_error(body.get("id"), -32601, f"Unknown method: {method}")
    params = body.get("params", {})
    message = params.get("message", {})
    parts = message.get("parts", [])
    if not parts:
        return _jsonrpc_error(body.get("id"), -32602, "No message parts")
    data = parts[0].get("data", {})
    if data.get("bootstrap"):
        return _handle_bootstrap(body.get("id"), data)
    return await _handle_turn(body.get("id"), data)


def _handle_bootstrap(request_id, data) -> JSONResponse:
    context_id = str(uuid.uuid4())
    benchmark_context = data.get("benchmark_context") or []
    tools = data.get("tools") or []
    system_prompt = _build_system_prompt(benchmark_context, tools)
    create_session(context_id, system_prompt, tools)
    logger.info("Bootstrap: context_id=%s tools=%d", context_id, len(tools))
    return _jsonrpc_success(request_id, {
        "kind": "data",
        "data": {"bootstrapped": True, "context_id": context_id},
    })


async def _handle_turn(request_id, data) -> JSONResponse:
    context_id = data.get("context_id")
    messages = data.get("messages") or []
    if not context_id or get_session(context_id) is None:
        return _jsonrpc_error(request_id, -32004, f"Unknown context_id: {context_id}")
    for msg in messages:
        add_message(context_id, msg)
    try:
        result_part = await _run_agent(context_id)
        return _jsonrpc_success(request_id, result_part)
    except Exception as exc:
        logger.exception("Agent failed")
        return _jsonrpc_error(request_id, -32000, str(exc))


def main() -> None:
    global _model, _seed, _card_url
    load_env()
    parser = argparse.ArgumentParser(description="pi-bench purple agent v2")
    parser.add_argument("--model", type=str, default="groq/llama-3.3-70b-versatile")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--card-url", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    _model = args.model
    _seed = args.seed
    _card_url = args.card_url
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("Starting purple agent v2: model=%s port=%d", _model, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()