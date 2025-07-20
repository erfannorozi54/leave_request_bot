import nest_asyncio

nest_asyncio.apply()
from tools.browser_tools import BrowserTools
import asyncio
import os, getpass
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
if "LLM_API_TOKEN" not in os.environ:
    os.environ["LLM_API_TOKEN"] = getpass.getpass("Enter your LLM API key: ")
api_key = os.getenv("LLM_API_TOKEN") or exit("LLM_API_TOKEN not set.")

# Load username and password from environment
username = os.getenv("CLIENT") or exit("USERNAME not set in environment.")
password = os.getenv("PASSWORD") or exit("PASSWORD not set in environment.")
print(username, password)
# 🎛️ Rate limiter: 1 request per minute
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1 / 15,  # ~0.0167 requests/second
    check_every_n_seconds=1,  # wake every second to refill
    max_bucket_size=1,  # max burst = 1
)

llm = ChatOpenAI(
    model="openai/gpt-4.1-nano",
    api_key=SecretStr(api_key),
    base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
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
            "You are a leave requesting robot for clercks on https://erp.asigi.net/hr/admin site you should go to this link and login ,to this site, username is '{username}' and password is '{password}'. then try to insert my leaves in this site. try to use all kind of tools to do the correct action."
        ),
        ("user", "{text}"),
    ]
)


async def main():
    browser = BrowserTools(headless=False)
    tools = browser.get_tools()
    memory = MemorySaver()
    agent = create_react_agent(llm, tools, checkpointer=memory)

    user_text = "insert my leaves on he site for 15 khordad, it is daily leave"
    prompt = prompt_template.invoke(
        {"username": username, "password": password, "text": user_text}
    )

    async for step in agent.astream(
        prompt, config={"configurable": {"thread_id": "abc22"}}, stream_mode="values"
    ):
        step["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
