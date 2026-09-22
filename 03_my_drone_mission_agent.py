import math
import sys

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)


@tool
def calculate_flight_time(
    distance_km: float,
    drone_speed_kmh: float,
    wind_speed_kmh: float,
    flight_bearing_deg: float,
    wind_direction_from_deg: float,
) -> float | str:
    """
    Calculate the flight time of a drone in minutes considering wind.

    Args:
        distance_km: Total flight distance in kilometres.
        drone_speed_kmh: Drone airspeed in km/h.
        wind_speed_kmh: Wind speed in km/h.
        flight_bearing_deg: Flight direction in degrees.
        wind_direction_from_deg: Direction FROM which the wind comes.
    """

    print(
        f"\n[TOOL EXECUTED] calculate_flight_time("
        f"distance_km={distance_km}, "
        f"drone_speed_kmh={drone_speed_kmh}, "
        f"wind_speed_kmh={wind_speed_kmh}, "
        f"flight_bearing_deg={flight_bearing_deg}, "
        f"wind_direction_from_deg={wind_direction_from_deg})"
    )

    if distance_km <= 0:
        return "Error: Distance must be greater than 0 km."

    if drone_speed_kmh <= 0:
        return "Error: Drone speed must be greater than 0 km/h."

    if wind_speed_kmh < 0:
        return "Error: Wind speed cannot be negative."

    flight_bearing_deg %= 360
    wind_direction_from_deg %= 360

    angle_rad = math.radians(
        wind_direction_from_deg - flight_bearing_deg
    )

    alpha = math.pi - angle_rad

    lateral_wind = wind_speed_kmh * math.sin(alpha)

    if abs(lateral_wind) >= drone_speed_kmh:
        return (
            "Error: The drone cannot compensate for the crosswind. "
            "The flight is unsafe."
        )

    ground_speed = (
        wind_speed_kmh * math.cos(alpha)
        + math.sqrt(
            drone_speed_kmh ** 2
            - lateral_wind ** 2
        )
    )

    if ground_speed <= 0:
        return (
            "Error: The headwind prevents the drone from moving "
            "forward along the route."
        )

    time_hours = distance_km / ground_speed
    time_minutes = time_hours * 60

    return round(time_minutes, 2)


@tool
def check_battery(
    flight_time_minutes: float,
    battery_percent: float,
    consumption_percent_per_minute: float,
    reserve_percent: float,
) -> str:
    """
    Check whether the drone has enough battery to complete the flight.

    Args:
        flight_time_minutes: Estimated flight time in minutes.
        battery_percent: Current drone battery percentage.
        consumption_percent_per_minute: Estimated battery consumption per minute.
        reserve_percent: Minimum battery percentage that must remain after landing.
    """

    print(
        f"\n[TOOL EXECUTED] check_battery("
        f"flight_time_minutes={flight_time_minutes}, "
        f"battery_percent={battery_percent}, "
        f"consumption_percent_per_minute={consumption_percent_per_minute}, "
        f"reserve_percent={reserve_percent})"
    )

    if flight_time_minutes <= 0:
        return "Error: Flight time must be greater than 0 minutes."

    if not 0 <= battery_percent <= 100:
        return "Error: Battery percentage must be between 0 and 100."

    if consumption_percent_per_minute <= 0:
        return "Error: Battery consumption per minute must be greater than 0."

    if not 0 <= reserve_percent <= 100:
        return "Error: Reserve percentage must be between 0 and 100."

    required_battery = (
        flight_time_minutes * consumption_percent_per_minute
    )

    battery_after_flight = battery_percent - required_battery

    if battery_after_flight < reserve_percent:
        return (
            f"Battery is insufficient. "
            f"The flight would consume approximately {required_battery:.2f}% "
            f"and {battery_after_flight:.2f}% would remain. "
            f"A minimum reserve of {reserve_percent}% is required."
        )

    return (
        f"Battery is sufficient. "
        f"The flight would consume approximately {required_battery:.2f}% "
        f"and {battery_after_flight:.2f}% would remain."
    )


@tool
def check_wind_safety(
    wind_speed_kmh: float,
    max_safe_wind_speed_kmh: float,
) -> str:
    """
    Check whether the current wind speed is safe for the drone.

    Args:
        wind_speed_kmh: Current wind speed in km/h.
        max_safe_wind_speed_kmh: Maximum safe wind speed supported by the drone.
    """

    print(
        f"\n[TOOL EXECUTED] check_wind_safety("
        f"wind_speed_kmh={wind_speed_kmh}, "
        f"max_safe_wind_speed_kmh={max_safe_wind_speed_kmh})"
    )

    if wind_speed_kmh < 0:
        return "Error: Wind speed cannot be negative."

    if max_safe_wind_speed_kmh <= 0:
        return "Error: Maximum safe wind speed must be greater than 0."

    if wind_speed_kmh > max_safe_wind_speed_kmh:
        return (
            f"Unsafe flight conditions. "
            f"The wind speed is {wind_speed_kmh} km/h "
            f"and the drone limit is {max_safe_wind_speed_kmh} km/h."
        )

    return (
        f"Wind conditions are acceptable. "
        f"The wind speed is {wind_speed_kmh} km/h "
        f"and the drone limit is {max_safe_wind_speed_kmh} km/h."
    )


drone_mission_agent = create_agent(
    model=llm,
    tools=[
        calculate_flight_time,
        check_battery,
        check_wind_safety,
    ],
    system_prompt=(
        "You are a drone mission specialist. "
        "Your job is to analyse whether a drone mission is feasible. "

        "You have tools to calculate flight time, check battery availability, "
        "and check wind safety. "

        "Use the available tools whenever the required information is available. "

        "Convert cardinal directions to degrees when necessary: "
        "North = 0 degrees, East = 90 degrees, South = 180 degrees, "
        "and West = 270 degrees. "
        "This conversion is deterministic and must not be considered an assumption. "

        "If the user provides a value clearly in natural language, extract and use it. "
        "Never invent values that the user has not provided. "

        "If information genuinely required by a tool is missing, ask the user for it. "

        "You may need to use the output of one tool as the input of another tool. "

        "When all necessary information is available, use all relevant tools "
        "and provide a final assessment of the mission. "

        "Respond in English."
    ),
)


print("Drone mission agent")

print(
    "\nDescribe the drone mission."
    "\nWrite 'exit' at any moment to close the program.\n"
)


while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ["exit", "quit", "salir"]:
        print("\nClosing Drone Mission Agent...")
        sys.exit()

    print("\n[AGENT] Thinking...")

    result = drone_mission_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        }
    )

    final_answer = result["messages"][-1].content

    print("\nAGENT:")
    print(final_answer)

    print("\nWould you like to ask anything else?")
    print("Type 'yes' to continue or 'no' to exit.")

    while True:
        continue_choice = input("You: ").strip().lower()

        if not continue_choice:
            continue

        if continue_choice in ["no", "n", "exit", "quit", "salir"]:
            print("\nClosing Drone Mission Agent...")
            sys.exit()

        if continue_choice in ["yes", "y", "si", "sí"]:
            print()
            break

        print("Please type 'yes' to continue or 'no' to exit.")