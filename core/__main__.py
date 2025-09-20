import sys
import dotenv
from typing import Dict, Any
import traceback

from langchain_core.messages import HumanMessage, AIMessage

from .agent_runtime import AgentRuntime

# Load environment variables from a .env file at the project root
dotenv.load_dotenv()


def _print_tool_results(state: Dict[str, Any]):
    """Helper to print tool results if they exist in the state."""
    tool_results = state.get("tool_results")
    if not tool_results:
        return
    
    print("\n" + "─" * 20 + " 🔧 TOOL RESULTS " + "─" * 20)
    for res in tool_results:
        tool_name = res.get("tool_name", "<unknown>")
        tool_output = str(res.get("result", ""))
        if "\n" in tool_output:
            print(f"[{tool_name}]\n---\n{tool_output}\n---")
        else:
            print(f"[{tool_name}] > {tool_output}")
    print("─" * 56)


def _print_token_usage(state: Dict[str, Any]):
    """Helper to print token usage details."""
    last_usage = state.get("last_token_usage", {}) or {}
    session_usage = state.get("session_token_usage", {}) or {}
    print(
        f"📊 Tokens (last): "
        f"In={last_usage.get('input', 0)}, "
        f"Out={last_usage.get('output', 0)}, "
        f"Total={last_usage.get('total', 0)}"
    )
    print(
        f"📈 Tokens (session): "
        f"In={session_usage.get('input', 0)}, "
        f"Out={session_usage.get('output', 0)}, "
        f"Total={session_usage.get('total', 0)} | "
        f"Turns={session_usage.get('count', 0)}"
    )


def _summarize_messages(prefix: str, messages: Any) -> None:
    try:
        recent = list(messages)[-5:]
        print(f"[DBG] {prefix}: total={len(messages)}")
        for i, m in enumerate(recent):
            mtype = getattr(m, "type", m.__class__.__name__)
            has_tools = bool(getattr(m, "tool_calls", []))
            preview = getattr(m, "content", "")
            if isinstance(preview, str):
                preview = preview.replace("\n", " ")
                if len(preview) > 80:
                    preview = preview[:77] + "..."
            print(f"[DBG]   {len(messages)-len(recent)+i}: type={mtype}, has_tool_calls={has_tools} | {preview}")
    except Exception:
        pass


def run_cli(agent_dir: str):
    """
    Main CLI loop that interacts with a specified AgentRuntime.
    """
    try:
        runtime = AgentRuntime(agent_dir)
        print("=" * 60)
        print(f"🤖 Agent '{runtime.config.get('name')}' loaded. Welcome back!")
        print(f"   Provider: {runtime.config.get('provider') or 'default'} | Type 'quit' to exit.")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Error loading agent from '{agent_dir}': {e}")
        return

    while True:
        try:
            user_input = input("\n📝 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            if not user_input:
                continue

            runtime.state["messages"].append(HumanMessage(content=user_input))
            _summarize_messages("pre-invoke messages", runtime.state.get("messages", []))

            print(f"\n🤖 {runtime.config.get('name')}: ", end="", flush=True)
            final_state = runtime.graph.invoke(runtime.state)
            _summarize_messages("post-invoke messages", final_state.get("messages", []))
            
            # Extract the last AI message from the clean final state
            last_ai_message = next((m for m in reversed(final_state.get("messages", [])) if isinstance(m, AIMessage)), None)
            response_text = last_ai_message.content if last_ai_message else "No response."
            print(response_text)
            # ----------------------------------------------------------------

            runtime.state = final_state
            runtime.save_state()  # This will now work correctly

            _print_tool_results(final_state)
            _print_token_usage(final_state)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            traceback.print_exc()
            try:
                _summarize_messages("on-exception messages", runtime.state.get("messages", []))
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli(sys.argv[1])
    else:
        # Corrected usage message for clarity
        print("Usage: python -m core <path_to_agent_directory>")
        print("Example: python -m core ./my-first-agent")

