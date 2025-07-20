import getpass
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import export_graph, stream_graph_updates


load_dotenv()

if not os.getenv("LLM_API_TOKEN"):
    os.environ["LLM_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
llm = init_chat_model(
    model="openai/gpt-4.1-nano",
    model_provider="openai",
    api_key=SecretStr(os.getenv("LLM_API_TOKEN") or ""),
    base_url=os.getenv("LLM_BASE_URL"),
)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
thread_id = "main_conversation"

config = RunnableConfig(configurable={"thread_id": thread_id})

if MemorySaver:
    checkpointer = MemorySaver()
    graph = graph_builder.compile(checkpointer=checkpointer)
    print("✅ Graph compiled with checkpointer")
else:
    graph = graph_builder.compile()
    print("⚠️  Graph compiled without checkpointer - state access limited")
# graph = graph_builder.compile()

g: Graph = graph.get_graph()


export_graph(g)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(graph, user_input, config)
    print(next(graph.get_state_history(config)))
    # I want to print the state of the graph
    # print(graph.get_state(config=config))