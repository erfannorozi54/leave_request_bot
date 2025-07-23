from tools.browser_tools import BrowserTools
import asyncio
import os, getpass
from dotenv import load_dotenv
from pydantic import SecretStr
from utils.utils import (
    load_llm_env_vars,
    State,
    export_graph,
    stream_until_done,
    pretty_print_messages,
)
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.graph import Graph


api_key, base_url, DEBUG = load_llm_env_vars()

# Load username and password from environment
username = os.getenv("USER_NAME") or exit("USERNAME not set in environment.")
password = os.getenv("PASSWORD") or exit("PASSWORD not set in environment.")

# 🎛️ Rate limiter: 1 request per minute
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1 / 15,  # ~0.0167 requests/second
    check_every_n_seconds=1,  # wake every second to refill
    max_bucket_size=1,  # max burst = 1
)

llm = ChatOpenAI(
    model="openai/gpt-4.1-nano",
    api_key=SecretStr(api_key),
    base_url=base_url,
    default_headers={
        "HTTP-Referer": "https://your-app.example",
        "X-Title": "Vacation-robot",
    },
    temperature=0.7,
    streaming=True,
    rate_limiter=rate_limiter,  # <-- add limiter here
)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an advanced leave requesting assistant designed to help clerks on the 'https://erp.asigi.net/hr/admin' portal. Your task is to log into the portal using the provided credentials, navigate the site, and efficiently submit leave requests. 

            To get started, navigate to 'https://erp.asigi.net/hr/admin' and log in using the username '{username}' and password '{password}'. After logging in, you should locate the leave request submission section and submit the leave. Throughout the process, you should leverage the available tools to assist with navigation, form filling, and content extraction.

            Tools available for your use:
            
            **Browser Tools**:
            - **Navigate to URL**: Use this to visit the portal or any webpage. Make sure to wait for the page to load before interacting with elements.
            - **Click Element**: Click on elements using CSS selectors. You can click buttons, links, or other interactive elements. Ensure that you click the correct element to proceed.
            - **Input Text**: Enter text into form fields by locating them via CSS selectors. This is essential for filling out the leave request form.
            - **Safe Click Element**: A more reliable version of the Click Element tool that waits for the element to become clickable before attempting to click it.
            - **Safe Input Text**: A more reliable version of the Input Text tool that waits for the element to be interactable before inputting text.
            - **Scroll**: Scroll the page vertically by a set number of pixels. Use this to view content that's not visible without scrolling.
            - **Wait for Element**: Wait for an element to appear on the page before proceeding with the next action. This ensures you interact with the right elements after page loads or dynamic changes.
            - **Get Page Content**: Retrieve all visible text from the current page. Useful for verifying information or confirming if the page has loaded as expected.
            - **Take Screenshot**: Capture a screenshot of the current page, which can be helpful for debugging or reporting.
            - **Press Key**: Simulate keyboard presses (like ENTER, TAB, ESC) to interact with form fields and buttons.
            - **Check Element Exists**: Verify if an element exists on the page, including its visibility and interactability. This helps avoid errors when interacting with elements.
            - **Find Elements by Text**: Search for elements that contain specific text. This can help locate form fields, buttons, or any text-based element.
            - **Get Page Info**: Collect detailed information about the page, including the number of elements like buttons, links, forms, and images. This is useful for understanding the page structure.
            - **Get Clickable Elements**: Discover all clickable elements on the page with details like their text, selector, and type. This helps identify which elements can be interacted with.
            - **Get Form Elements**: Retrieve all form elements on the page (inputs, buttons, textareas) with details about their names, IDs, and other attributes. This is helpful for identifying the fields you need to fill out for leave requests.
            - **Refresh Page**: Refresh the current page if needed, and wait for the page to reload.
            - **Go Back**: Navigate back to the previous page if needed.
            - **Go Forward**: Navigate forward in browser history if applicable.

            Your objective is to complete the leave request process by interacting with the site effectively, ensuring the leave request is submitted correctly. Be sure to use all tools at your disposal to perform these tasks with accuracy and efficiency.
            """,
        ),
        ("user", "{text}"),
    ]
)
browser_tools = BrowserTools(headless=False)
tools = browser_tools.get_tools()

tool_node = ToolNode(tools=tools)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # if isinstance(message, AIMessage):
    #     assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder = StateGraph(State)

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
    snapshot = graph.get_state(config)
    user_input = input("Enter your message: ")
    prompt = prompt_template.invoke(
        {"username": username, "password": password, "text": user_input}
    ).to_messages()
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    else:
        for event in stream_until_done(graph, {"messages": prompt}, config, DEBUG=DEBUG):
            # print(event["messages"][-1].content)
            pretty_print_messages(event, DEBUG=DEBUG)
