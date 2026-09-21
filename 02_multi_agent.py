from dotenv import load_dotenv
from langchain.agents import create_agent

# Import the tool decorator.
# The decorator exposes subagent wrapper functions as tools for the supervisor.
from langchain.tools import tool

load_dotenv()

# Weather tool
def get_weather(city: str) -> str:
    """Get weather information for a given city."""

    print(f"[TOOL EXECUTED] get_weather(city={city})")

    # Simulated weather response.
    return f"The weather in {city} is sunny, 24°C, with wind at 8 km/h."


# Weather agent
weather_agent = create_agent(
    model="openai:gpt-5.5",

    # This agent only has access to the weather tool.
    tools=[get_weather],

    # This prompt defines the specialization of this agent.
    system_prompt=(
        "You are a weather specialist. "
        "Answer weather-related questions using the available weather tool."
    ),
)


# Flight tool
def calculate_flight_time(distance_km: float, speed_kmh: float) -> str:
    # The docstring describes the tool behavior and its intended use cases.
    """Calculate drone flight time in minutes from distance in km and speed in km/h."""

    print(
        f"[TOOL EXECUTED] calculate_flight_time("
        f"distance_km={distance_km}, speed_kmh={speed_kmh})"
    )

    if speed_kmh <= 0:
        return "The speed must be greater than 0 km/h."
    
    time_hours = distance_km / speed_kmh
    time_minutes = time_hours * 60

    return (
        f"A drone travelling {distance_km} km at {speed_kmh} km/h "
        f"would take approximately {time_minutes:.2f} minutes."
    )


# Flight agent
flight_agent = create_agent(
    model="openai:gpt-5.5",

    # This agent only has access to calculate_flight_time, it does not know about get_weather.
    tools=[calculate_flight_time],

    system_prompt=(
        "You are a drone flight specialist. "
        "Use the available tool to calculate drone flight times."
    ),
)


# Two independent agents are configured:

# weather_agent
#      │
#      └── get_weather()

# flight_agent
#      │
#      └── calculate_flight_time()




# Coordination between agents:
# They are completely independent.

# weather_agent does not know flight_agent exists.
# flight_agent does not know weather_agent exists.

# Agent selection is handled automatically.
# An additional agent coordinates the specialized agents.
# This coordinating agent is the supervisor.




# Wrap weather agent as a tool for the supervisor agent.
@tool
def ask_weather_agent(query: str) -> str:
    """Delegate weather-related questions to the weather specialist agent."""

    print("\n[SUBAGENT CALLED] weather_agent")
    print(f"[SUBAGENT INPUT] {query}")

    # Subagent invocation flow.
    
    # Weather retrieval is delegated to the weather agent.
    # The wrapper invokes that agent.
    
    # The flow is:
    
    # supervisor
    #     │
    #     ▼
    # ask_weather_agent()
    #     │
    #     ▼
    # weather_agent.invoke()
    #     │
    #     ▼
    # weather agent LLM
    #     │
    #     ▼
    # get_weather()
    
    result = weather_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    # The weather agent may internally have several messages:

    # UserMessage
    # AIMessage with tool call
    # ToolMessage
    # AIMessage with final answer
    # Only the final response is exposed to the supervisor.

    # Return the final answer produced by the weather agent.
    return result["messages"][-1].content




# Wrap flight agent as a tool for the supervisor agent.
@tool
def ask_flight_agent(query: str) -> str:
    """Delegate drone flight-time calculations to the flight specialist agent."""

    print("\n[SUBAGENT CALLED] flight_agent")
    print(f"[SUBAGENT INPUT] {query}")

    result = flight_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    return result["messages"][-1].content


# Purpose of the @tool decorator

# The tools argument of create_agent() accepts callable tool definitions.
# weather_agent and flight_agent itself is an agent.

# The supervisor cannot simply receive:
# tools=[weather_agent, flight_agent]
# Wrapper functions expose the agents as callable tools:
# ask_weather_agent(query)
# and:
# ask_flight_agent(query)
# Internally those functions execute the corresponding agents.




# So conceptually:

# SUPERVISOR
#     │
#     ├── TOOL: ask_weather_agent()
#     │                 │
#     │                 ▼
#     │            weather_agent
#     │                 │
#     │                 ▼
#     │            get_weather()
#     │
#     └── TOOL: ask_flight_agent()
#                       │
#                       ▼
#                  flight_agent
#                       │
#                       ▼
#              calculate_flight_time()




# Supervisor agent configuration.
supervisor_agent = create_agent(
    model="openai:gpt-5.5",

    # Supervisor tool configuration.

    # The supervisor does NOT directly receive:
    # get_weather
    # calculate_flight_time
    
    # The supervisor receives wrapper functions for the two specialized agents
    # exposed as tools.
    tools=[
        ask_weather_agent,
        ask_flight_agent,
    ],

    system_prompt=(
        "You are a supervisor agent that coordinates specialized agents. "
        "Use ask_weather_agent for weather-related questions. "
        "Use ask_flight_agent for drone flight-time calculations. "
        "If the user asks for information requiring both specialists, "
        "delegate to both and combine their results into one final answer."
    ),
)




# User request
result = supervisor_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I want to fly a drone in Barcelona. "
                    "Tell me the weather and calculate how long it would take "
                    "to fly 12 km at 40 km/h."
                )
            }
        ]
    }
)




# The final answer is produced by the supervisor agent.
print("\nFINAL ANSWER:")
print(result["messages"][-1].content)


# Execution flow:

# The user asks:

# "Request: weather in Barcelona and flight time
# for a drone travelling
# 12 km at 40 km/h."

# The first model that receives this question is NOT:
# weather_agent

# and it is NOT:
# flight_agent

# It is:
# supervisor_agent



#  Step 1 - Supervisor sees the user request and decides what to do.
# Conceptually the supervisor sees:

# SYSTEM:
# System instruction: coordinate specialized agents.

# AVAILABLE TOOLS:

# ask_weather_agent(query: string)
# "Delegate weather-related questions to the weather specialist agent."

# ask_flight_agent(query: string)
# "Delegate drone flight-time calculations to the flight specialist agent."

# USER:
# Request: weather in Barcelona and flight time
# for a drone travelling
# 12 km at 40 km/h.

# The request contains two tasks:
# Both tasks require specialist input.
# 1. Weather information.
# 2. Flight-time calculation.
# Therefore it should delegate work to BOTH specialized agents.




# STEP 2 - Supervisor calls weather subagent
# The supervisor may generate something conceptually like:
# {
#     "name": "ask_weather_agent",
#     "arguments": {
#         "query": "What is the weather in Barcelona?"
#     }
# }

# Tool-call execution:
# The LLM generates instructions without executing Python.

# The supervisor model generates the tool-call instruction.
# LangChain sees that instruction.
# Python executes:
# ask_weather_agent(
#     query="What is the weather in Barcelona?"
# )



# STEP 3 - A second agent starts

# Inside ask_weather_agent(): weather_agent.invoke(...)
# starts another agent execution.

# Execution flow:

# SUPERVISOR LLM
#       │
#       │ requests ask_weather_agent()
#       ▼
# LANGCHAIN
#       │
#       ▼
# PYTHON FUNCTION
# ask_weather_agent()
#       │
#       ▼
# WEATHER AGENT
#       │
#       ▼
# WEATHER AGENT LLM




# STEP 4 - Weather agent calls its own tool

# The weather agent sees:

# USER:
# What is the weather in Barcelona?

# AVAILABLE TOOL:
# get_weather(city)

# It can generate:
# {
#     "name": "get_weather",
#     "arguments": {
#         "city": "Barcelona"
#     }
# }
#
# LangChain executes:
# get_weather("Barcelona")

# Python returns:
# "The weather in Barcelona is sunny,
# 24°C, with wind at 8 km/h."

# That information goes back to the WEATHER AGENT LLM.
# The weather agent generates its final response.

# ask_weather_agent() then returns that final response to the SUPERVISOR.




# STEP 5 - Flight subagent

# The supervisor also needs the flight calculation.
# So the same pattern happens:

# SUPERVISOR
#      │
#      ▼
# ask_flight_agent()
#      │
#      ▼
# flight_agent.invoke()
#      │
#      ▼
# FLIGHT AGENT LLM
#      │
#      ▼
# calculate_flight_time(
#     distance_km=12,
#     speed_kmh=40
# )

# Python calculates:
# time = distance / speed
# time = 12 / 40
# time = 0.3 hours
# 0.3 * 60 = 18 minutes

# So the tool returns approximately:
# "A drone travelling 12 km at 40 km/h
# would take approximately 18.00 minutes."




# STEP 6 - Results return to supervisor
# The supervisor now conceptually has:
# WEATHER SUBAGENT RESULT:
# "The weather in Barcelona is sunny..."

# FLIGHT SUBAGENT RESULT:
# "The flight would take approximately 18 minutes."

# The supervisor LLM sees those results
# and generates ONE final answer for the user.



# Centralized coordination

# Important:
# weather_agent does NOT decide to contact flight_agent.
# flight_agent does NOT decide to contact weather_agent.

# They do not communicate directly.

# The SUPERVISOR is the central coordinator.

#                   SUPERVISOR
#                    /       \
#                   /         \
#                  ▼           ▼
#           weather_agent   flight_agent

# This architecture is therefore centralized.




# Subagent context isolation

# The following invocation:

# weather_agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": query
#             }
#         ]
#     }
# )

# creates a new execution for the subagent.
# It does NOT automatically receive the entire conversation
# that the supervisor has had with the user.

# The subagent receives the explicit query supplied in:
# "content": query

# This is called context isolation.

# It is useful because a specialized agent does not necessarily
# need to know everything that has happened in the main conversation.

# More advanced systems can explicitly control
# what context each subagent receives.


#The MAIN result contains the state of the SUPERVISOR AGENT.

# This:
# result = supervisor_agent.invoke(...)
# contains the state of the SUPERVISOR AGENT.

# Conceptually:
# result
# │
# └── messages
#       │
#       ├── UserMessage
#       │
#       ├── AIMessage with subagent tool calls
#       │
#       ├── ToolMessage containing weather agent result
#       │
#       ├── ToolMessage containing flight agent result
#       │
#       └── AIMessage with final answer

# Message history scope:

# The internal message histories of: weather_agent and: flight_agent
# are NOT automatically copied into this main result.

# The supervisor receives the values returned by the wrapper functions:
# return result["messages"][-1].content

# So the internal work is isolated.


# LLM call sequence
# Multi-agent execution can involve more LLM calls than the single-agent example.

# A possible execution could be:
# SUPERVISOR LLM
#      │
#      │ decides to call weather agent
#      ▼
# WEATHER AGENT LLM
#      │
#      │ decides to use get_weather
#      ▼
# get_weather()
#      │
#      ▼
# WEATHER AGENT LLM
#      │
#      │ final weather result
#      ▼
# SUPERVISOR

# Then:
# SUPERVISOR
#      │
#      │ calls flight agent
#      ▼
# FLIGHT AGENT LLM
#      │
#      │ decides to calculate flight time
#      ▼
# calculate_flight_time()
#      │
#      ▼
# FLIGHT AGENT LLM
#      │
#      │ final flight result
#      ▼
# SUPERVISOR LLM
#      │
#      ▼
# FINAL ANSWER

# Therefore one user request can cause MULTIPLE LLM calls.
# This is also one reason why multi-agent systems can consume significantly more tokens/resources
# than a simple single-agent application.



# Complete flow diagram:

# 02_multi_agent.py
#          │
#          ▼
# load_dotenv()
#          │
#          ▼
# OPENAI_API_KEY loaded
#          │
#          ▼
# CREATE WEATHER AGENT
#          │
#          └── get_weather()
#
# CREATE FLIGHT AGENT
#          │
#          └── calculate_flight_time()
#
#          │
#          ▼
# Wrap both agents as tools
#          │
#          ├── ask_weather_agent()
#          └── ask_flight_agent()
#          │
#          ▼
# CREATE SUPERVISOR AGENT
#          │
#          ├── weather subagent tool
#          └── flight subagent tool
#          │
#          ▼
# supervisor_agent.invoke()
#          │
#          ▼
# USER REQUEST
#          │
#          ▼
# SUPERVISOR LLM
#          │
#          ├─────────────────────────┐
#          │                         │
#          ▼                         ▼
# ask_weather_agent()         ask_flight_agent()
#          │                         │
#          ▼                         ▼
# weather_agent              flight_agent
#          │                         │
#          ▼                         ▼
# WEATHER LLM                 FLIGHT LLM
#          │                         │
#          ▼                         ▼
# get_weather()       calculate_flight_time()
#          │                         │
#          ▼                         ▼
# WEATHER RESULT              FLIGHT RESULT
#          │                         │
#          └────────────┬────────────┘
#                       │
#                       ▼
#                 SUPERVISOR LLM
#                       │
#                       ▼
#                  FINAL ANSWER
#                       │
#                       ▼
#       result["messages"][-1].content