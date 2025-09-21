import sys
import dotenv
import json
from typing import Dict, Any, List, Optional
import traceback

from .agent_runtime import AgentRuntime

# Load environment variables from a .env file at the project root
dotenv.load_dotenv()


def _print_tool_results(tool_results: Optional[List[Dict[str, Any]]]):
    """Helper to print tool results if they exist."""
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


def _track_and_print_token_usage(runtime: AgentRuntime):
    """Prints the session's token usage details."""
    token_metrics = runtime.monitor.get_token_metrics()
    session_usage = token_metrics.get("session_total", {})

    print(
        f"📈 Tokens (session): "
        f"In={session_usage.get('input', 0)}, "
        f"Out={session_usage.get('output', 0)}, "
        f"Total={session_usage.get('total', 0)}"
    )


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
        traceback.print_exc()
        return

    while True:
        try:
            user_input = input("\n📝 You: ").strip()

            if not user_input:
                continue

            # --- META COMMANDS ---
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nShutting down and archiving session...")
                shutdown_summary = runtime.shutdown()
                print(shutdown_summary["message"])
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == "!analyze":
                print("\n🔬 Analyzing knowledge base...")
                analysis = runtime.analyze_knowledge_base()
                print(json.dumps(analysis, indent=2))
                continue
            
            # --- AGENT INVOCATION ---
            print(f"\n🤖 {runtime.config.get('name')}: ", end="", flush=True)
            
            turn_output = runtime.invoke(user_input)
            
            print(turn_output["response"])

            _print_tool_results(turn_output["tool_results"])
            _track_and_print_token_usage(runtime)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            runtime.monitor.log_event("error", {"message": str(e), "traceback": traceback.format_exc()})
            print(f"\n❌ An unexpected error occurred: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli(sys.argv[1])
    else:
        print("Usage: python -m core <path_to_agent_directory>")
        print("Example: python -m core ./my-first-agent")

