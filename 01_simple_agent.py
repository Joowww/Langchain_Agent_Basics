from dotenv import load_dotenv
from langchain.agents import create_agent


# Load variables from the local .env file
load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    print(f"[TOOL EXECUTED] get_weather(city={city})")

    return f"It's always sunny in {city}!"


agent = create_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in San Francisco?"
            }
        ]
    }
)


print("\nFINAL ANSWER:")
print(result["messages"][-1].content)