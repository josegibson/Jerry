from __future__ import annotations

import os
from operator import add
from typing import List, Optional, Any, Literal, TypedDict, Annotated

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .agent_monitor import AgentMonitor

# Optional model backends
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

# ==============================================================================
# 1. STATE DEFINITION
# ==============================================================================
class ConversationState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add]
    provider_name: Optional[Literal["OpenAI", "Gemini"]]
    retrieved_context: str

# ==============================================================================
# 2. GRAPH BUILDER CLASS
# ==============================================================================
class GraphBuilder:
    def __init__(self, llm: Any, tools: List[BaseTool], retriever: Any, system_prompt: str, 
                 monitor: AgentMonitor, provider_name: str, memory_window: int = 10):
        self.llm_with_tools = llm.bind_tools(tools)
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.tools = tools
        self.monitor = monitor
        self.provider_name = provider_name
        self.memory_window = memory_window

    def _summarize_messages_for_log(self, msgs: List[BaseMessage]) -> List[Dict[str, Any]]:
        summary = []
        for m in msgs[-10:]:
            role = getattr(m, "type", getattr(m, "role", m.__class__.__name__))
            content_preview = getattr(m, "content", "")
            if isinstance(content_preview, str):
                content_preview = content_preview.replace("\n", " ")
                if len(content_preview) > 120:
                    content_preview = content_preview[:117] + "..."
            summary.append({
                "role": role,
                "has_tool_calls": bool(getattr(m, "tool_calls", None)),
                "content_preview": content_preview
            })
        return summary

    def node_retrieve(self, state: ConversationState) -> dict:
        retrieved_context = ""
        last_user_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        if last_user_msg:
            docs = self.retriever.invoke(last_user_msg)
            context = "\n\n".join(f"Source: {d.metadata.get('source', 'N/A')}\n{d.page_content}" for d in docs)
            retrieved_context = context
            self.monitor.log_event("graph_debug", {
                "node": "retrieve",
                "message": "Retrieved context for user message.",
                "last_user_msg_len": len(last_user_msg),
                "retrieved_docs": len(docs),
                "retrieved_context_len": len(retrieved_context)
            })
        return {"retrieved_context": retrieved_context}

    def node_llm(self, state: ConversationState) -> dict:
        messages_to_send = [SystemMessage(content=self.system_prompt)]
        if state.get("retrieved_context"):
            messages_to_send.append(SystemMessage(content=f"Use the following context to answer the user:\n{state['retrieved_context']}"))
        messages_to_send.extend(state["messages"][-self.memory_window:])
        
        self.monitor.log_event("graph_debug", {
            "node": "llm", "sub_event": "pre_invoke",
            "message": f"Sending {len(messages_to_send)} messages to LLM.",
            "messages_summary": self._summarize_messages_for_log(messages_to_send)
        })
        
        try:
            response = self.llm_with_tools.invoke(messages_to_send)
            
            # --- TOKEN TRACKING ---
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.monitor.track_token_usage(
                    provider_name=self.provider_name,
                    usage_data=response.usage_metadata
                )
            # --- END TOKEN TRACKING ---

            self.monitor.log_event("graph_debug", {
                "node": "llm", "sub_event": "post_invoke",
                "message": "Received response from LLM.",
                "response_type": response.__class__.__name__,
                "has_tool_calls": bool(getattr(response, 'tool_calls', None))
            })

        except Exception as e:
            self.monitor.log_event("graph_error", {
                "node": "llm", "message": f"LLM invocation failed: {e}",
                "failed_payload_summary": self._summarize_messages_for_log(messages_to_send)
            })
            raise
            
        return {"messages": [response]}

    def route_from_llm(self, state: ConversationState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        has_tool_calls = bool(getattr(last_message, "tool_calls", []))
        self.monitor.log_event("graph_debug", {
            "node": "router",
            "message": f"Routing based on LLM response. Has tool calls: {has_tool_calls}",
            "last_message_type": last_message.__class__.__name__,
        })
        return "tools" if has_tool_calls else "__end__"

    def build(self) -> Any:
        graph = StateGraph(ConversationState)
        graph.add_node("retrieve", self.node_retrieve)
        graph.add_node("llm", self.node_llm)
        tool_node = ToolNode(self.tools)
        graph.add_node("tools", tool_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "llm")
        graph.add_conditional_edges(
            "llm",
            self.route_from_llm,
            {"tools": "tools", "__end__": END}
        )
        graph.add_edge("tools", "llm")
        self.monitor.log_event("graph_debug", {"message": "Graph compiled successfully."})
        return graph.compile()

# ==============================================================================
# 3. PUBLIC FACTORY FUNCTION
# ==============================================================================

def create_agent_graph(
    tools: List[BaseTool],
    retriever: Any,
    system_prompt: str,
    monitor: AgentMonitor,
    preferred_provider: Optional[str] = None,
    memory_window: int = 10
) -> Any:
    preferred = (preferred_provider or "").lower()
    llm = None

    providers = {
        'gemini': lambda: ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2),
        'openai': lambda: ChatOpenAI(model="gpt-4-turbo", temperature=0.2, streaming=True)
    }
    
    available_providers = ['gemini', 'openai']
    if preferred in available_providers:
        available_providers.insert(0, available_providers.pop(available_providers.index(preferred)))

    selected_provider = None
    for provider_name in available_providers:
        try:
            llm = providers[provider_name]()
            selected_provider = provider_name
            if llm:
                monitor.log_event("graph_debug", {"message": f"Successfully initialized LLM provider: {provider_name}"})
                break
        except Exception as ex:
            monitor.log_event("graph_warning", {"message": f"Provider init failed for {provider_name}: {ex}"})
            continue

    if not llm:
        raise RuntimeError("No LLM provider configured. Set OPENAI_API_KEY or GOOGLE_API_KEY.")

    monitor.log_event("graph_debug", {"message": f"Using provider: {selected_provider}"})
    builder = GraphBuilder(
        llm=llm,
        tools=tools,
        retriever=retriever,
        system_prompt=system_prompt,
        monitor=monitor,
        provider_name=selected_provider,
        memory_window=memory_window
    )
    return builder.build()

