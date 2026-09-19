# This imports the functions from the dotenv package to load environment variables from a .env file.
from dotenv import load_dotenv
# This imports the create_agent function from the langchain.agents module to create an AI agent.
from langchain.agents import create_agent


# Load variables from the local .env file
load_dotenv()


def get_weather(city: str) -> str:
    # Normally docstrings describe what a function does for developers.
    # But with agents, it has another important function: 
    # it tells the agent what the tool does, so it can decide when to use it.
    # The model needs to know:
    # Tool name: get_weather
    # Tool description: Get weather information for a given city
    # Parameters: city (str): The name of the city to get the weather for.
    """Get weather for a given city."""

    # Not important, only to see when the tool is executed in the console.
    print(f"[TOOL EXECUTED] get_weather(city={city})")

    # The return, as it can be seen its fake.
    return f"It's always sunny in {city}!"


agent = create_agent(
    # Specify the model to use.
    # LangChain interprets that and internally creates the appropriate model integration.
    # Instead of manually doing something like: client = OpenAI(...)
    # LangChain handle that abstraction.
    # The order is: your model input -> LangChain recognizes provider -> langchain-openai integration -> OpenAI API call.
    model="openai:gpt-5.5",

    # Gives the agent acces to the function (it could be more functions inside []), so it can call it when needed.
    tools=[get_weather],

    # Sets the system prompt for the agent to shape how the agent approaches tasks.
    system_prompt="You are a helpful assistant",
)

# Invoke roughly means run this agent once with this input.
result = agent.invoke(
    {
        # LangChain agents keep their conversational input in a messages state.
        # The messages sredescribed as the conversation history field used by the agent.
        "messages": [
            {
                # The role of the message sender, in this case, the user.
                # Important because conversations with LLMs consist of different kinds of messages.
                # In this case, it represents the human talking to the agent.
                "role": "user",
                "content": "What's the weather in San Francisco?"
            }
        ]
    }
)



# So to understand the flow what enters the agent is approximately:

# SYSTEM:
# You are a helpful assistant

# AVAILABLE TOOL:
# get_weather(city: string)
# "Get weather for a given city."

# USER:
# What's the weather in San Francisco?




# What does the LLM see?

# Conceptually, LangChain prepares information telling the model:
# You are a helpful assistant.

# You have the following tool available:

# get_weather
# Description:
# Get weather for a given city.

# Input:
# city: string

# User:
# What's the weather in San Francisco?

# Then the LLM needs to decide if it needs to call the tool or not. 
# If it does, it will call the tool and then return the result to the user.




# The tool call

# The model may generate something conceptually like:

# {
#   "name": "get_weather",
#   "arguments": {
#     "city": "San Francisco"
#   }
# }

# CRUCIAL TO KNOW that the model is not executing Python.
# The model only generates a structured instruction saying:
# Call get_weather with city="San Francisco".
# Then LangChain sees that tool request.




# Who actually runs the function?

# The flow becomes:

# GPT
#  │
#  │ "Call get_weather"
#  ▼
# LangChain
#  │
#  │ get_weather(city="San Francisco")
#  ▼
# Python

# Then python executes: get_weather("San Francisco")
# The line of the print appear: [TOOL EXECUTED] get_weather(city=San Francisco)
# And returns the result: "It's always sunny in San Francisco!"




# The result goes back to the model
# The tool result does not necessarily become the final response directly.

# Instead of:

# Tool
#  │
#  │ "It's always sunny in San Francisco!"
#  ▼
# LangChain
#  │
#  ▼
# LLM again

# The model now has more context:

# USER:
# What's the weather in San Francisco?

# ASSISTANT:
# I need get_weather.

# TOOL:
# It's always sunny in San Francisco!

# Then the model generates the final natural-language answer.




# One question may call the LLM more than once

# You might think:
# 1 question = 1 LLM call

# But with an agent it may be:
# LLM
#  ↓
# tool A
#  ↓
# LLM
#  ↓
# tool B
#  ↓
# LLM
#  ↓
# tool C
#  ↓
# LLM
#  ↓
# final answer

# In line 44 we have result but what is result?
# When: result = agent.invoke(...), finishes, result contains the final agent state.
# It isn't just: "It's sunny."
# It contains data including the messages generated during execution.

# LangChain documents the built-in messages state as an append-only conversation history
# Conceptually:

# result
# │
# └── messages
#       │
#       ├── UserMessage
#       │
#       ├── AIMessage with tool call
#       │
#       ├── ToolMessage
#       │
#       └── AIMessage with final answer

# This is going to be very useful because in the next step we'll inspect those messages.



print("\nFINAL ANSWER:")
print(result["messages"][-1].content)

# Why [-1]?

# Because -1 means: the last element of the list
# And as we know, Python list indexing works like:
# 0 first element
# 1 second element
# 2 third element
# ...
# -1 last element

# So: result["messages"][-1] means that:
# Give me the last message generated by the agent.

# That should be the final assistant response.



# .content

# It retrieves the content of that final message
# So: print(result["messages"][-1].content), means that:
# Take the final message, extract its content, and print it.

# Full execution diagram of the most basic core of an AI agent:

# 01_simple_agent.py
#        │
#        ▼
# load_dotenv()
#        │
#        ▼
# OPENAI_API_KEY loaded
#        │
#        ▼
# create_agent()
#        │
#        ├── model = GPT-5.5
#        ├── tool = get_weather
#        └── system prompt
#        │
#        ▼
# agent.invoke()
#        │
#        ▼
# USER MESSAGE
# "What's the weather in San Francisco?"
#        │
#        ▼
# OPENAI MODEL
#        │
#        │ decides:
#        │ "I should use get_weather"
#        ▼
# TOOL CALL
# get_weather(
#     city="San Francisco"
# )
#        │
#        ▼
# PYTHON FUNCTION EXECUTES
#        │
#        ▼
# "It's always sunny in San Francisco!"
#        │
#        ▼
# TOOL RESULT
# sent back to agent/model
#        │
#        ▼
# OPENAI MODEL
#        │
#        ▼
# FINAL RESPONSE
#        │
#        ▼
# result["messages"][-1].content