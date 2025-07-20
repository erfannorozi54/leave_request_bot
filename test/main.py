import getpass
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

if not os.getenv("LLM_API_TOKEN"):
    os.environ["LLM_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")


llm = ChatOpenAI(
    model="openai/gpt-4.1-nano",
    api_key=SecretStr(os.getenv("LLM_API_TOKEN") or ""),
    base_url=os.getenv("LLM_BASE_URL"),
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke({"messages": [{"role": "user", "content": "what is the weather in Tabriz"}]})
