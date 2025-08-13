from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal, Tuple, TypedDict
import os

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END

from common.prompts import TOOL_GUIDANCE, ASSISTANT_SYSTEM_PROMPT
from vault_tools.tools import get_vault_tools
from vault_tools.vectorstore import get_retriever

# Optional model backends
try:
    from langchain_openai import ChatOpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False


class ConversationState(TypedDict, total=False):
    messages: List[BaseMessage]
    tool_results: List[Dict[str, Any]]
    provider_name: Optional[Literal["OpenAI", "Gemini"]]
    # Token usage accounting
    last_token_usage: Dict[str, int]
    session_token_usage: Dict[str, int]
    # Retrieved context
    retrieved_context: str


# ----- Model selection -----
def _select_chat_model(preferred: Optional[str] = None) -> Tuple[Any, str]:
    preferred = (preferred or "").lower()

    # Try explicit preference first
    if preferred == "openai":
        if not _HAS_OPENAI:
            raise RuntimeError("OpenAI backend not available (missing langchain-openai)")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        return (
            ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7),
            "OpenAI",
        )

    if preferred == "gemini":
        if not _HAS_GEMINI:
            raise RuntimeError("Gemini backend not available (missing langchain-google-genai)")
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is not set")
        return (
            ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.7),
            "Gemini",
        )

    # Auto-detect (Gemini first if present, then OpenAI)
    if _HAS_GEMINI and os.getenv("GEMINI_API_KEY"):
        return (
            ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.7),
            "Gemini",
        )
    if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        return (
            ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7),
            "OpenAI",
        )

    raise RuntimeError(
        "No LLM backend available. Install langchain-openai or langchain-google-genai and set API keys."
    )


def _select_with_fallback(preferred: Optional[str] = None) -> Tuple[Any, str]:
    """Try to select the preferred provider; on error, print and switch to the other."""
    try:
        return _select_chat_model(preferred)
    except Exception as primary_error:
        print(f"[Provider error] {primary_error}. Attempting fallback provider...")
        # Flip provider explicitly to avoid repeating the same failure
        preferred_norm = (preferred or "").lower()
        fallback_pref = "openai" if preferred_norm == "gemini" else ("gemini" if preferred_norm == "openai" else "openai")
        try:
            return _select_chat_model(fallback_pref)
        except Exception as fallback_error:
            print(f"[Fallback failed] {fallback_error}")
            raise RuntimeError(
                f"Failed to initialize any provider. Primary error: {primary_error}. Fallback error: {fallback_error}"
            )


def _extract_token_usage(ai_message: AIMessage) -> Dict[str, int]:
    """Best-effort extraction of token usage from an AIMessage into a normalized dict.

    Returns a dict with integer keys: input, output, total. Missing values default to 0.
    """
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    # 1) usage_metadata is standardized where available
    usage_meta = getattr(ai_message, "usage_metadata", None) or {}
    if isinstance(usage_meta, dict):
        input_tokens = int(usage_meta.get("input_tokens", input_tokens) or 0)
        output_tokens = int(usage_meta.get("output_tokens", output_tokens) or 0)
        total_tokens = int(usage_meta.get("total_tokens", total_tokens) or (input_tokens + output_tokens))

    # 2) Provider-specific response metadata fallbacks
    resp_meta = getattr(ai_message, "response_metadata", None) or {}
    if isinstance(resp_meta, dict):
        # OpenAI (python): token_usage = {prompt_tokens, completion_tokens, total_tokens}
        token_usage = resp_meta.get("token_usage") or resp_meta.get("tokenUsage")
        if isinstance(token_usage, dict):
            input_tokens = max(input_tokens, int(token_usage.get("prompt_tokens", token_usage.get("promptTokens", 0)) or 0))
            output_tokens = max(output_tokens, int(token_usage.get("completion_tokens", token_usage.get("completionTokens", 0)) or 0))
            # total may be present; otherwise compute
            total_tokens = max(total_tokens, int(token_usage.get("total_tokens", token_usage.get("totalTokens", 0)) or (input_tokens + output_tokens)))
        # Anthropic-like: usage = {input_tokens, output_tokens}
        usage = resp_meta.get("usage")
        if isinstance(usage, dict):
            input_tokens = max(input_tokens, int(usage.get("input_tokens", 0) or 0))
            output_tokens = max(output_tokens, int(usage.get("output_tokens", 0) or 0))
            total_tokens = max(total_tokens, input_tokens + output_tokens)

    return {"input": input_tokens, "output": output_tokens, "total": total_tokens}


# ----- Nodes -----

def node_retrieve(state: ConversationState) -> ConversationState:
    state.setdefault("messages", [])
    # Get last user query
    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content
            break
    if not last_user:
        return state

    # Build retriever on persisted Chroma
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    retriever = get_retriever(persist_directory=persist_dir, top_k=int(os.getenv("RETRIEVE_TOP_K", "5")))
    results = retriever.invoke(last_user)

    # Concise context block with sources
    context_lines: List[str] = []
    for idx, doc in enumerate(results, start=1):
        source = doc.metadata.get("source") or doc.metadata.get("path") or doc.metadata.get("file_path") or "unknown"
        context_lines.append(f"[{idx}] {source}\n{doc.page_content}")
    context_text = "\n\n".join(context_lines) if context_lines else ""
    state["retrieved_context"] = context_text
    return state


def node_llm(state: ConversationState, preferred_provider: Optional[str], tools: List[BaseTool]) -> ConversationState:
    # Ensure required keys exist
    state.setdefault("messages", [])
    state.setdefault("tool_results", [])
    state.setdefault("session_token_usage", {"input": 0, "output": 0, "total": 0})

    # Gather short history excluding any previous SystemMessage we added
    prior_history = [m for m in state["messages"] if not isinstance(m, SystemMessage)]

    # Find last user message content
    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content
            break

    if not last_user:
        return state

    # Select chat model (with fallback) and bind tools
    chat_model, provider_name = _select_with_fallback(preferred_provider)
    llm_with_tools = chat_model.bind_tools(tools)

    # Build working messages: persona, tool guidance, retrieved context, then history
    working_msgs: List[BaseMessage] = [
        SystemMessage(content=ASSISTANT_SYSTEM_PROMPT),
        SystemMessage(content=TOOL_GUIDANCE),
    ]
    retrieved = state.get("retrieved_context")
    if retrieved:
        working_msgs.append(SystemMessage(content=f"Relevant context from the user's vault:\n\n{retrieved}"))
    working_msgs.extend(prior_history)

    try:
        ai_msg = llm_with_tools.invoke(working_msgs)
    except Exception as invoke_error:
        print(f"[Provider runtime error with {provider_name}] {invoke_error}. Switching provider...")
        # Try alternate provider once
        alt_pref = "openai" if provider_name == "Gemini" else "gemini"
        alt_model, alt_provider = _select_with_fallback(alt_pref)
        llm_with_tools = alt_model.bind_tools(tools)
        ai_msg = llm_with_tools.invoke(working_msgs)
        provider_name = alt_provider

    # Normalize and account for token usage
    if isinstance(ai_msg, AIMessage):
        usage = _extract_token_usage(ai_msg)
    else:
        usage = {"input": 0, "output": 0, "total": 0}

    state["last_token_usage"] = usage
    sess = state.get("session_token_usage", {"input": 0, "output": 0, "total": 0})
    sess["input"] = int(sess.get("input", 0)) + int(usage.get("input", 0))
    sess["output"] = int(sess.get("output", 0)) + int(usage.get("output", 0))
    sess["total"] = int(sess.get("total", 0)) + int(usage.get("total", 0))
    sess['count'] = int(sess.get('count', 0)) + 1
    state["session_token_usage"] = sess

    state["messages"].append(ai_msg if isinstance(ai_msg, AIMessage) else AIMessage(content=str(ai_msg)))
    state["provider_name"] = provider_name
    return state


def node_tools(state: ConversationState, tools: List[BaseTool]) -> ConversationState:
    state.setdefault("messages", [])
    state.setdefault("tool_results", [])

    if not state["messages"]:
        return state

    last_ai = state["messages"][-1]
    tool_calls = getattr(last_ai, "tool_calls", None)
    if not tool_calls:
        return state

    # Execute each requested tool and append ToolMessages
    for idx, call in enumerate(tool_calls, start=1):
        tool_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        tool_args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        tool_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)

        result = f"Tool '{tool_name}' not found"
        for t in tools:
            if t.name == tool_name:
                try:
                    result = t.invoke(tool_args)
                except Exception as e:
                    result = f"Error executing tool {tool_name}: {e}"
                break

        state["tool_results"].append({"tool_name": tool_name, "result": result})
        state["messages"].append(ToolMessage(content=str(result), tool_call_id=tool_id or f"call_{idx}"))

    return state


# ----- Routing helpers -----
def _route_from_llm(state: ConversationState):
    if not state.get("messages"):
        return END
    last_ai = state["messages"][-1]
    has_tools = hasattr(last_ai, "tool_calls") and bool(getattr(last_ai, "tool_calls", None))
    return "tools" if has_tools else END


# ----- Graph builder -----
def build_conversation_graph(preferred_provider: Optional[str] = None):
    tools: List[BaseTool] = get_vault_tools()

    graph = StateGraph(ConversationState)
    graph.add_node("retrieve", lambda s: node_retrieve(s))
    graph.add_node("llm", lambda s: node_llm(s, preferred_provider, tools))
    graph.add_node("tools", lambda s: node_tools(s, tools))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "llm")
    graph.add_conditional_edges("llm", _route_from_llm, {"tools": "tools", END: END})
    graph.add_edge("tools", "retrieve")

    return graph.compile()