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
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from utils.utils import export_graph, State
from langchain_core.messages import HumanMessage, AIMessage


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

search_tool = TavilySearch(max_results=2)


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    print("before interrupt")
    human_response = interrupt({"query": query})
    print(f"Not pauses and waits for human response")
    return human_response


tools = [search_tool, human_assistance]

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    if isinstance(message, AIMessage):
        assert len(message.tool_calls) <= 1
    return {"messages": [message]}


tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
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

pending_interrupt = False

while True:
    if pending_interrupt:
        # If we are in an interrupted state, resume the graph
        user_answer = input("🧑‍💻 Human reply: ")
        for event in graph.stream(
            Command(resume=user_answer),
            config=config,
            stream_mode="values",
        ):
            if "messages" in event:
                event["messages"][-1].pretty_print()
    else:
        # Otherwise, get new user input
        user_input = input("User: ")
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        # Run the graph
        for event in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values",
        ):
            print(f"DEBUG: {event}")
            if "messages" in event and isinstance(event["messages"][-1], AIMessage) and event["messages"][-1].tool_calls:
                print("---Tool Call---")
                event["messages"][-1].pretty_print()
            elif "messages" in event:
                event["messages"][-1].pretty_print()

    # Always check for interruptions after every graph execution
    pending_interrupt = False
    state = graph.get_state(config)
    if state.next and "tools" in state.next:
        last_message = state.values["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "human_assistance":
                    pending_interrupt = True
                    break
