import getpass
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import (
    export_graph,
    stream_graph_updates,
    BasicToolNode,
    State,
    route_tools,
)


load_dotenv()

if not os.getenv("LLM_API_TOKEN"):
    os.environ["LLM_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")
if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API key: ")


graph_builder = StateGraph(State)
llm = init_chat_model(
    model="openai/gpt-4.1-nano",
    model_provider="openai",
    api_key=SecretStr(os.getenv("LLM_API_TOKEN") or ""),
    base_url=os.getenv("LLM_BASE_URL"),
)

tool = TavilySearch(max_results=2)
tools = [tool]

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)

thread_id = "main_conversation"
config = RunnableConfig(configurable={"thread_id": thread_id})

if MemorySaver:
    checkpointer = MemorySaver()
    graph = graph_builder.compile(checkpointer=checkpointer)
    print("✅ Graph compiled with checkpointer")
else:
    graph = graph_builder.compile()
    print("⚠️  Graph compiled without checkpointer - state access limited")

g: Graph = graph.get_graph()


export_graph(g)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    graph = stream_graph_updates(graph, user_input, config)
    # I want to print the state of the graph
    # print(graph.get_state(config=config))
