from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal, Tuple, TypedDict
import os

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END

from common.prompts import TOOL_GUIDANCE
from vault_tools.tools import get_vault_tools

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


# ----- Nodes -----
def node_llm(state: ConversationState, preferred_provider: Optional[str], tools: List[BaseTool]) -> ConversationState:
    # Ensure required keys exist
    state.setdefault("messages", [])
    state.setdefault("tool_results", [])

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

    # Select chat model and bind tools
    chat_model, provider_name = _select_chat_model(preferred_provider)
    llm_with_tools = chat_model.bind_tools(tools)

    # Invoke once; if tool calls are present, a tools node will handle them
    working_msgs: List[BaseMessage] = [SystemMessage(content=TOOL_GUIDANCE), *prior_history, HumanMessage(content=last_user)]
    ai_msg = llm_with_tools.invoke(working_msgs)

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
    graph.add_node("llm", lambda s: node_llm(s, preferred_provider, tools))
    graph.add_node("tools", lambda s: node_tools(s, tools))

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", _route_from_llm, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()