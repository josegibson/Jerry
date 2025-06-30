"""
Main CLI application logic (LangGraph-native)
"""
import os
from langchain_core.messages import HumanMessage, AIMessage
from .graph import build_conversation_graph, ConversationState


class MultiLLMCLI:
    def __init__(self):
        self.history = []  # Stores LangChain message objects for context
        self.graph_app = build_conversation_graph()
        # Initialize dict-style state
        self.graph_state: ConversationState = {
            "messages": list(self.history),
            "tool_results": [],
            "provider_name": None,
        }

    def run(self):
        """Main CLI loop (LangGraph)"""
        print("\n" + "="*50)
        print("🤖 LangGraph Chat")
        print("="*50)
        print("Type your questions/prompts below.")
        print("Commands: 'quit', 'exit', or 'q' to exit")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n📝 You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Append to conversation and invoke graph
                self.history.append(HumanMessage(content=user_input))
                self.graph_state["messages"].append(HumanMessage(content=user_input))

                updated_state = self.graph_app.invoke(self.graph_state)
                ai_messages = [m for m in updated_state["messages"] if isinstance(m, AIMessage)]
                response_text = ai_messages[-1].content if ai_messages else ""

                print(f"\n🤖 {updated_state.get('provider_name') or 'LLM'}: ", end="", flush=True)
                print(response_text)

                if updated_state.get("tool_results"):
                    print("\n🔧 Tool results:")
                    for res in updated_state["tool_results"]:
                        tool_name = res.get("tool_name", "<unknown>")
                        tool_output = res.get("result", "")
                        print(f"\n[{tool_name}]\n{tool_output}")

                # Sync local state
                self.graph_state = updated_state
                self.history.append(AIMessage(content=response_text))

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}") 