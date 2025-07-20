from pathlib import Path
from langchain_core.runnables.graph import Graph
import json
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables.config import RunnableConfig
from typing import Annotated, TypedDict, Union
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.constants import END


def export_graph(
    g: Graph,
    json_path: Path = Path("my_graph.json"),
    png_path: Path = Path("my_graph.png"),
) -> None:
    """
    Export a graph to a JSON file and a PNG file.
    """
    # 1. Export JSON
    json_str = json.dumps(g.to_json(), ensure_ascii=False, indent=2)
    json_path.write_text(json_str, encoding="utf-8")
    print(f"✅  Wrote JSON to {json_path.resolve()}")

    # 2. Render Mermaid PNG bytes
    png_bytes = g.draw_mermaid_png()

    # 3. Save PNG
    png_path.write_bytes(png_bytes)
    print(f"✅  Wrote PNG to {png_path.resolve()}")


def stream_graph_updates(
    graph: CompiledStateGraph, user_input: str, config: RunnableConfig
) -> CompiledStateGraph:
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]}, config=config
    ):
        for value in event.values():
            if value.get("messages", []):
                last_msg = value["messages"][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    print("Assistant:", last_msg.content)
    return graph


class State(TypedDict):
    messages: Annotated[list[Union[BaseMessage, ToolMessage, AIMessage]], add_messages]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State) -> State:
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        else:
            raise ValueError("The message is not an AIMessage")
        return {"messages": outputs}


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the LLM.
    """
    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if (
        hasattr(ai_message, "tool_calls")
        and isinstance(ai_message, AIMessage)
        and len(ai_message.tool_calls) > 0
    ):
        return "tools"
    else:
        return END
