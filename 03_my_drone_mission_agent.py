import math
import re
from dataclasses import dataclass

from langchain.tools import tool
from langchain_ollama import ChatOllama


@dataclass
class MissionState:
    distance_km: float | None = None
    drone_speed_kmh: float | None = None
    wind_speed_kmh: float | None = None
    flight_bearing_deg: float | None = None
    wind_direction_from_deg: float | None = None
    battery_percent: float | None = None
    consumption_percent_per_minute: float | None = None
    reserve_percent: float | None = None
    max_safe_wind_speed_kmh: float | None = None


DIRECTIONS = {
    "north": 0.0,
    "n": 0.0,
    "northeast": 45.0,
    "ne": 45.0,
    "east": 90.0,
    "e": 90.0,
    "southeast": 135.0,
    "se": 135.0,
    "south": 180.0,
    "s": 180.0,
    "southwest": 225.0,
    "sw": 225.0,
    "west": 270.0,
    "w": 270.0,
    "northwest": 315.0,
    "nw": 315.0,
}


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    reasoning=False,
)


@tool
def calculate_flight_time(
    distance_km: float,
    drone_speed_kmh: float,
    wind_speed_kmh: float,
    flight_bearing_deg: float,
    wind_direction_from_deg: float,
) -> dict:
    """
    Calculate drone flight time considering wind.
    """

    print(
        "\n[TOOL EXECUTED] calculate_flight_time("
        f"distance_km={distance_km}, "
        f"drone_speed_kmh={drone_speed_kmh}, "
        f"wind_speed_kmh={wind_speed_kmh}, "
        f"flight_bearing_deg={flight_bearing_deg}, "
        f"wind_direction_from_deg={wind_direction_from_deg})"
    )

    if distance_km <= 0:
        return {
            "status": "error",
            "message": "Distance must be greater than 0 km.",
        }

    if drone_speed_kmh <= 0:
        return {
            "status": "error",
            "message": "Drone speed must be greater than 0 km/h.",
        }

    if wind_speed_kmh < 0:
        return {
            "status": "error",
            "message": "Wind speed cannot be negative.",
        }

    flight_bearing_deg %= 360
    wind_direction_from_deg %= 360

    relative_angle_rad = math.radians(
        wind_direction_from_deg - flight_bearing_deg
    )

    headwind_component_kmh = (
        wind_speed_kmh * math.cos(relative_angle_rad)
    )

    crosswind_component_kmh = (
        wind_speed_kmh * math.sin(relative_angle_rad)
    )

    if abs(crosswind_component_kmh) >= drone_speed_kmh:
        return {
            "status": "unsafe",
            "message": "The drone cannot compensate for the crosswind.",
            "headwind_component_kmh": round(
                headwind_component_kmh,
                2,
            ),
            "crosswind_component_kmh": round(
                crosswind_component_kmh,
                2,
            ),
        }

    ground_speed_kmh = (
        -headwind_component_kmh
        + math.sqrt(
            drone_speed_kmh**2
            - crosswind_component_kmh**2
        )
    )

    if ground_speed_kmh <= 0:
        return {
            "status": "unsafe",
            "message": (
                "The headwind prevents the drone from moving "
                "forward along the route."
            ),
            "ground_speed_kmh": round(
                ground_speed_kmh,
                2,
            ),
            "headwind_component_kmh": round(
                headwind_component_kmh,
                2,
            ),
            "crosswind_component_kmh": round(
                crosswind_component_kmh,
                2,
            ),
        }

    flight_time_minutes = (
        distance_km / ground_speed_kmh
    ) * 60

    return {
        "status": "success",
        "flight_time_minutes": round(
            flight_time_minutes,
            2,
        ),
        "ground_speed_kmh": round(
            ground_speed_kmh,
            2,
        ),
        "headwind_component_kmh": round(
            headwind_component_kmh,
            2,
        ),
        "crosswind_component_kmh": round(
            crosswind_component_kmh,
            2,
        ),
    }


@tool
def check_wind_safety(
    wind_speed_kmh: float,
    max_safe_wind_speed_kmh: float,
) -> dict:
    """
    Check whether the wind speed is safe for the drone.
    """

    print(
        "\n[TOOL EXECUTED] check_wind_safety("
        f"wind_speed_kmh={wind_speed_kmh}, "
        f"max_safe_wind_speed_kmh={max_safe_wind_speed_kmh})"
    )

    if wind_speed_kmh < 0:
        return {
            "status": "error",
            "message": "Wind speed cannot be negative.",
        }

    if max_safe_wind_speed_kmh <= 0:
        return {
            "status": "error",
            "message": (
                "Maximum safe wind speed must be greater than 0."
            ),
        }

    wind_safe = (
        wind_speed_kmh <= max_safe_wind_speed_kmh
    )

    return {
        "status": "safe" if wind_safe else "unsafe",
        "wind_safe": wind_safe,
        "wind_speed_kmh": round(
            wind_speed_kmh,
            2,
        ),
        "max_safe_wind_speed_kmh": round(
            max_safe_wind_speed_kmh,
            2,
        ),
    }


@tool
def check_battery(
    flight_time_minutes: float,
    battery_percent: float,
    consumption_percent_per_minute: float,
    reserve_percent: float,
) -> dict:
    """
    Check whether the drone has enough battery for the flight.
    """

    print(
        "\n[TOOL EXECUTED] check_battery("
        f"flight_time_minutes={flight_time_minutes}, "
        f"battery_percent={battery_percent}, "
        f"consumption_percent_per_minute="
        f"{consumption_percent_per_minute}, "
        f"reserve_percent={reserve_percent})"
    )

    if flight_time_minutes <= 0:
        return {
            "status": "error",
            "message": "Flight time must be greater than 0.",
        }

    if not 0 <= battery_percent <= 100:
        return {
            "status": "error",
            "message": "Battery percentage must be between 0 and 100.",
        }

    if consumption_percent_per_minute <= 0:
        return {
            "status": "error",
            "message": (
                "Battery consumption per minute must be greater than 0."
            ),
        }

    if not 0 <= reserve_percent <= 100:
        return {
            "status": "error",
            "message": "Reserve percentage must be between 0 and 100.",
        }

    required_battery_percent = (
        flight_time_minutes
        * consumption_percent_per_minute
    )

    battery_after_flight_percent = (
        battery_percent
        - required_battery_percent
    )

    battery_sufficient = (
        battery_after_flight_percent
        >= reserve_percent
    )

    return {
        "status": (
            "safe"
            if battery_sufficient
            else "unsafe"
        ),
        "battery_sufficient": battery_sufficient,
        "required_battery_percent": round(
            required_battery_percent,
            2,
        ),
        "battery_after_flight_percent": round(
            battery_after_flight_percent,
            2,
        ),
        "required_reserve_percent": round(
            reserve_percent,
            2,
        ),
    }


def to_float(value: str) -> float:
    return float(
        value.replace(",", ".")
    )


def extract_direction(
    text: str,
    pattern: str,
) -> float | None:
    match = re.search(
        pattern,
        text,
        re.IGNORECASE,
    )

    if not match:
        return None

    direction = (
        match.group(1)
        .lower()
        .replace("-", "")
        .replace(" ", "")
    )

    return DIRECTIONS.get(direction)


def save_value(
    state: MissionState,
    field_name: str,
    value: float,
    updated_fields: set[str],
) -> None:
    setattr(
        state,
        field_name,
        value,
    )

    updated_fields.add(
        field_name
    )


def update_state(
    state: MissionState,
    user_input: str,
) -> set[str]:
    text = user_input.lower()

    updated_fields = set()

    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*"
        r"(?:km|kilometres?|kilometers?)"
        r"(?!\s*/?\s*h)",
        text,
        re.IGNORECASE,
    )

    if match:
        save_value(
            state,
            "distance_km",
            to_float(match.group(1)),
            updated_fields,
        )

    speed_patterns = [
        r"(?:travelling|traveling|flying|moving)"
        r"\s+at\s+"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*km\s*/?\s*h",

        r"drone(?:\s+speed|\s+airspeed)"
        r"[^0-9]{0,20}"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*km\s*/?\s*h",
    ]

    for pattern in speed_patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            save_value(
                state,
                "drone_speed_kmh",
                to_float(match.group(1)),
                updated_fields,
            )
            break

    match = re.search(
        r"(?:the\s+)?wind(?:\s+speed)?"
        r"\s+(?:is|at|of)?\s*"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*km\s*/?\s*h",
        text,
        re.IGNORECASE,
    )

    if match:
        save_value(
            state,
            "wind_speed_kmh",
            to_float(match.group(1)),
            updated_fields,
        )

    match = re.search(
        r"(?:maximum|max)\s+safe\s+wind\s+speed"
        r"(?:\s+for\s+the\s+drone)?"
        r"\s*(?:is|=|of)?\s*"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*km\s*/?\s*h",
        text,
        re.IGNORECASE,
    )

    if match:
        save_value(
            state,
            "max_safe_wind_speed_kmh",
            to_float(match.group(1)),
            updated_fields,
        )

    battery_patterns = [
        r"(?:the\s+)?drone\s+"
        r"(?:currently\s+)?has\s+"
        r"(\d+(?:[.,]\d+)?)\s*%"
        r"\s+battery"
        r"(?:\s+(?:remaining|left|available))?",

        r"(\d+(?:[.,]\d+)?)\s*%"
        r"\s+battery\s+"
        r"(?:remaining|left|available)",

        r"(?:current\s+)?battery"
        r"(?:\s+(?:level|percentage))?"
        r"\s*(?:is|at|=)\s*"
        r"(\d+(?:[.,]\d+)?)\s*%",
    ]

    for pattern in battery_patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            save_value(
                state,
                "battery_percent",
                to_float(match.group(1)),
                updated_fields,
            )
            break

    consumption_patterns = [
        r"(?:consumes?|consumption"
        r"(?:\s+rate)?(?:\s+is)?)"
        r"[^0-9]{0,30}"
        r"(\d+(?:[.,]\d+)?)\s*%"
        r"(?:\s+battery)?"
        r"\s*(?:per\s+minute|/\s*min)",

        r"(\d+(?:[.,]\d+)?)\s*%"
        r"(?:\s+battery)?"
        r"\s*(?:per\s+minute|/\s*min)",
    ]

    for pattern in consumption_patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            save_value(
                state,
                "consumption_percent_per_minute",
                to_float(match.group(1)),
                updated_fields,
            )
            break

    reserve_patterns = [
        r"(\d+(?:[.,]\d+)?)\s*%"
        r"(?:\s+battery)?"
        r"\s+reserve",

        r"(?:reserve|minimum\s+reserve)"
        r"[^0-9]{0,30}"
        r"(\d+(?:[.,]\d+)?)\s*%",
    ]

    for pattern in reserve_patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            save_value(
                state,
                "reserve_percent",
                to_float(match.group(1)),
                updated_fields,
            )
            break

    bearing_match = re.search(
        r"(?:bearing|heading|flight\s+direction)"
        r"[^0-9]{0,20}"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*(?:degrees|degree|deg|°)",
        text,
        re.IGNORECASE,
    )

    if bearing_match:
        save_value(
            state,
            "flight_bearing_deg",
            to_float(
                bearing_match.group(1)
            ) % 360,
            updated_fields,
        )

    else:
        direction = extract_direction(
            text,
            r"\b(?:fly|flying)\b.*?"
            r"\b(?:km|kilometres?|kilometers?)\b\s+"
            r"(north(?:east|west)?|"
            r"south(?:east|west)?|"
            r"east|west|ne|nw|se|sw|n|s|e|w)\b",
        )

        if direction is None:
            direction = extract_direction(
                text,
                r"\b(?:towards?|heading|direction)\s+"
                r"(north(?:east|west)?|"
                r"south(?:east|west)?|"
                r"east|west|ne|nw|se|sw|n|s|e|w)\b",
            )

        if direction is not None:
            save_value(
                state,
                "flight_bearing_deg",
                direction,
                updated_fields,
            )

    wind_bearing_match = re.search(
        r"wind.*?\bfrom\s+(?:the\s+)?"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*(?:degrees|degree|deg|°)",
        text,
        re.IGNORECASE,
    )

    if wind_bearing_match:
        save_value(
            state,
            "wind_direction_from_deg",
            to_float(
                wind_bearing_match.group(1)
            ) % 360,
            updated_fields,
        )

    else:
        wind_direction = extract_direction(
            text,
            r"wind.*?\bfrom\s+(?:the\s+)?"
            r"(north(?:east|west)?|"
            r"south(?:east|west)?|"
            r"east|west|ne|nw|se|sw|n|s|e|w)\b",
        )

        if wind_direction is not None:
            save_value(
                state,
                "wind_direction_from_deg",
                wind_direction,
                updated_fields,
            )

    return updated_fields


def format_trace(
    trace: list[str],
) -> str:
    return "\n".join(
        f"{number}. {step}"
        for number, step in enumerate(
            trace,
            start=1,
        )
    )


def analyse_mission(
    state: MissionState,
) -> dict:
    flight_result = None
    wind_result = None
    battery_result = None

    trace = []

    trace.append(
        "Read the current mission parameters."
    )

    flight_data_available = all(
        value is not None
        for value in [
            state.distance_km,
            state.drone_speed_kmh,
            state.wind_speed_kmh,
            state.flight_bearing_deg,
            state.wind_direction_from_deg,
        ]
    )

    if flight_data_available:
        trace.append(
            "All parameters required for the flight dynamics "
            "analysis are available."
        )

        flight_result = calculate_flight_time.invoke(
            {
                "distance_km":
                    state.distance_km,
                "drone_speed_kmh":
                    state.drone_speed_kmh,
                "wind_speed_kmh":
                    state.wind_speed_kmh,
                "flight_bearing_deg":
                    state.flight_bearing_deg,
                "wind_direction_from_deg":
                    state.wind_direction_from_deg,
            }
        )

        relative_angle = (
            state.wind_direction_from_deg
            - state.flight_bearing_deg
        )

        trace.append(
            "Calculate the relative wind angle: "
            f"{state.wind_direction_from_deg}° - "
            f"{state.flight_bearing_deg}° = "
            f"{relative_angle}°."
        )

        if "headwind_component_kmh" in flight_result:
            trace.append(
                "Headwind component: "
                f"{flight_result['headwind_component_kmh']} km/h."
            )

        if "crosswind_component_kmh" in flight_result:
            trace.append(
                "Crosswind component: "
                f"{flight_result['crosswind_component_kmh']} km/h."
            )

        if flight_result["status"] == "success":
            trace.append(
                "The drone can compensate for the crosswind."
            )

            trace.append(
                "Ground speed after wind correction: "
                f"{flight_result['ground_speed_kmh']} km/h."
            )

            trace.append(
                "Calculate flight time: "
                f"{state.distance_km} km / "
                f"{flight_result['ground_speed_kmh']} km/h × 60 = "
                f"{flight_result['flight_time_minutes']} minutes."
            )

        else:
            if (
                "crosswind_component_kmh"
                in flight_result
                and abs(
                    flight_result[
                        "crosswind_component_kmh"
                    ]
                )
                >= state.drone_speed_kmh
            ):
                trace.append(
                    "Compare the crosswind magnitude with the "
                    "drone airspeed: "
                    f"{abs(flight_result['crosswind_component_kmh'])} "
                    f"km/h >= {state.drone_speed_kmh} km/h."
                )

            trace.append(
                flight_result["message"]
            )

    else:
        trace.append(
            "Flight dynamics cannot be calculated because one "
            "or more required parameters are missing."
        )

    if (
        state.wind_speed_kmh is not None
        and state.max_safe_wind_speed_kmh is not None
    ):
        wind_result = check_wind_safety.invoke(
            {
                "wind_speed_kmh":
                    state.wind_speed_kmh,
                "max_safe_wind_speed_kmh":
                    state.max_safe_wind_speed_kmh,
            }
        )

        trace.append(
            "Compare the wind speed with the drone safety limit: "
            f"{state.wind_speed_kmh} km/h versus "
            f"{state.max_safe_wind_speed_kmh} km/h."
        )

        if wind_result["status"] == "safe":
            trace.append(
                "The wind speed is within the drone's general "
                "safe operating limit."
            )

        else:
            trace.append(
                "The wind speed exceeds the drone's general "
                "safe operating limit."
            )

    else:
        trace.append(
            "Wind safety cannot be determined because required "
            "information is missing."
        )

    battery_parameters_available = all(
        value is not None
        for value in [
            state.battery_percent,
            state.consumption_percent_per_minute,
            state.reserve_percent,
        ]
    )

    if (
        flight_result is not None
        and flight_result.get("status") == "success"
        and battery_parameters_available
    ):
        battery_result = check_battery.invoke(
            {
                "flight_time_minutes":
                    flight_result["flight_time_minutes"],
                "battery_percent":
                    state.battery_percent,
                "consumption_percent_per_minute":
                    state.consumption_percent_per_minute,
                "reserve_percent":
                    state.reserve_percent,
            }
        )

        trace.append(
            "Calculate required battery: "
            f"{flight_result['flight_time_minutes']} minutes × "
            f"{state.consumption_percent_per_minute}%/min = "
            f"{battery_result['required_battery_percent']}%."
        )

        trace.append(
            "Calculate battery after flight: "
            f"{state.battery_percent}% - "
            f"{battery_result['required_battery_percent']}% = "
            f"{battery_result['battery_after_flight_percent']}%."
        )

        trace.append(
            "Compare the remaining battery with the required "
            f"{state.reserve_percent}% reserve."
        )

        if battery_result["status"] == "safe":
            trace.append(
                "The remaining battery satisfies the required reserve."
            )

        else:
            trace.append(
                "The remaining battery does not satisfy the "
                "required reserve."
            )

    elif (
        flight_result is not None
        and flight_result.get("status") != "success"
    ):
        trace.append(
            "Battery analysis is not performed because the flight "
            "dynamics are already unsafe and no valid flight time "
            "is available."
        )

    elif not battery_parameters_available:
        trace.append(
            "Battery safety cannot yet be fully calculated because "
            "one or more battery parameters are missing."
        )

    else:
        trace.append(
            "Battery safety cannot be calculated because a valid "
            "flight time is not available."
        )

    output = []
    unsafe = False

    if flight_result is None:
        output.append(
            "Flight calculation: UNDETERMINED"
        )

    elif flight_result["status"] == "success":
        output.append(
            "Estimated flight time: "
            f"{flight_result['flight_time_minutes']} minutes"
        )

        output.append(
            "Estimated ground speed: "
            f"{flight_result['ground_speed_kmh']} km/h"
        )

        output.append(
            "Headwind component: "
            f"{flight_result['headwind_component_kmh']} km/h"
        )

        output.append(
            "Crosswind component: "
            f"{flight_result['crosswind_component_kmh']} km/h"
        )

    else:
        unsafe = True

        output.append(
            "Flight dynamics: UNSAFE"
        )

        output.append(
            flight_result["message"]
        )

    if wind_result is None:
        output.append(
            "Wind safety: UNDETERMINED"
        )

    elif wind_result["status"] == "safe":
        output.append(
            "Wind safety: SAFE "
            f"({wind_result['wind_speed_kmh']} km/h <= "
            f"{wind_result['max_safe_wind_speed_kmh']} km/h)"
        )

    else:
        unsafe = True

        output.append(
            "Wind safety: UNSAFE "
            f"({wind_result['wind_speed_kmh']} km/h > "
            f"{wind_result['max_safe_wind_speed_kmh']} km/h)"
        )

    if battery_result is None:
        output.append(
            "Battery safety: UNDETERMINED"
        )

    else:
        if battery_result["status"] == "safe":
            output.append(
                "Battery safety: SAFE"
            )

        else:
            unsafe = True

            output.append(
                "Battery safety: UNSAFE"
            )

        output.append(
            "Estimated battery consumption: "
            f"{battery_result['required_battery_percent']}%"
        )

        output.append(
            "Estimated battery after flight: "
            f"{battery_result['battery_after_flight_percent']}%"
        )

        output.append(
            "Required battery reserve: "
            f"{battery_result['required_reserve_percent']}%"
        )

    if unsafe:
        output.append(
            "\nFinal mission feasibility: NOT FEASIBLE"
        )

        trace.append(
            "At least one safety condition failed, therefore the "
            "mission is NOT FEASIBLE."
        )

        return {
            "report": "\n".join(output),
            "trace": format_trace(trace),
        }

    missing = []

    if state.distance_km is None:
        missing.append(
            "flight distance"
        )

    if state.drone_speed_kmh is None:
        missing.append(
            "drone speed"
        )

    if state.wind_speed_kmh is None:
        missing.append(
            "wind speed"
        )

    if state.flight_bearing_deg is None:
        missing.append(
            "flight direction or bearing"
        )

    if state.wind_direction_from_deg is None:
        missing.append(
            "wind direction"
        )

    if state.max_safe_wind_speed_kmh is None:
        missing.append(
            "maximum safe wind speed"
        )

    if state.battery_percent is None:
        missing.append(
            "current battery percentage"
        )

    if (
        state.consumption_percent_per_minute
        is None
    ):
        missing.append(
            "battery consumption percentage per minute"
        )

    if state.reserve_percent is None:
        missing.append(
            "required battery reserve"
        )

    if missing:
        output.append(
            "\nMissing information:"
        )

        for item in missing:
            output.append(
                f"- {item}"
            )

        output.append(
            "\nFinal mission feasibility: UNDETERMINED"
        )

        trace.append(
            "Mission feasibility remains UNDETERMINED because "
            "required information is still missing."
        )

    else:
        output.append(
            "\nFinal mission feasibility: FEASIBLE"
        )

        trace.append(
            "Flight dynamics, wind safety and battery safety all "
            "passed. The mission is FEASIBLE."
        )

    return {
        "report": "\n".join(output),
        "trace": format_trace(trace),
    }


def is_trace_request(
    user_input: str,
) -> bool:
    text = user_input.lower()

    phrases = [
        "all the steps",
        "show me the steps",
        "show the steps",
        "explain the steps",
        "steps you follow",
        "steps that you follow",
        "how did you calculate",
        "how did you get",
        "how did you reach",
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


def is_analysis_request(
    user_input: str,
) -> bool:
    text = user_input.lower()

    phrases = [
        "is the mission feasible",
        "is this mission feasible",
        "check the mission",
        "analyse the mission",
        "analyze the mission",
        "recalculate",
        "recompute",
        "check feasibility",
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


def state_to_text(
    state: MissionState,
) -> str:
    return (
        f"distance_km={state.distance_km}\n"
        f"drone_speed_kmh={state.drone_speed_kmh}\n"
        f"wind_speed_kmh={state.wind_speed_kmh}\n"
        f"flight_bearing_deg={state.flight_bearing_deg}\n"
        f"wind_direction_from_deg={state.wind_direction_from_deg}\n"
        f"battery_percent={state.battery_percent}\n"
        f"consumption_percent_per_minute="
        f"{state.consumption_percent_per_minute}\n"
        f"reserve_percent={state.reserve_percent}\n"
        f"max_safe_wind_speed_kmh="
        f"{state.max_safe_wind_speed_kmh}"
    )


def clean_llm_response(
    content: str,
) -> str:
    content = re.sub(
        r"<think>.*?</think>\s*",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if "</think>" in content.lower():
        parts = re.split(
            r"</think>",
            content,
            flags=re.IGNORECASE,
            maxsplit=1,
        )

        if len(parts) == 2:
            content = parts[1]

    return content.strip()


def answer_question(
    user_input: str,
    state: MissionState,
    last_analysis: dict | None,
) -> str:
    if last_analysis is None:
        report = "No mission has been analysed yet."
        trace = "No calculation steps are available yet."

    else:
        report = last_analysis["report"]
        trace = last_analysis["trace"]

    system_prompt = f"""
You are a drone mission assistant.

Answer the user's question using only the mission state,
analysis results and calculation trace provided below.

Never invent or modify mission parameters.
Never invent calculation results.
Do not perform new safety calculations yourself.
If information is unavailable, say that it is unavailable.

Do not reveal hidden reasoning or internal thinking.
Only provide the final concise explanation to the user.

For crosswind signs, use the exact convention used by this program:

relative_angle = wind_direction_from_deg - flight_bearing_deg

crosswind_component =
wind_speed * sin(relative_angle)

Therefore, for a drone flying east at 90 degrees with wind
coming from north at 0 degrees:

relative_angle = 0 - 90 = -90 degrees

sin(-90 degrees) = -1

so a 12 km/h wind produces a crosswind component of -12 km/h.

The magnitude represents the crosswind strength.
The sign represents its direction according to this mathematical
convention.

CURRENT MISSION STATE:
{state_to_text(state)}

LAST ANALYSIS:
{report}

CALCULATION TRACE:
{trace}

Respond in English.
"""

    response = llm.invoke(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                user_input,
            ),
        ],
        reasoning=False,
    )

    return clean_llm_response(
        response.content
    )


def main() -> None:
    state = MissionState()
    last_analysis = None

    print("Drone Mission Agent")

    print(
        "\nDescribe the drone mission."
        "\nYou can provide missing information in later messages."
        "\nYou can also ask questions about the current mission."
        "\nWrite 'reset' to start a new mission."
        "\nWrite 'exit' to close the program."
    )

    while True:
        try:
            user_input = input(
                "\nYou: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print(
                "\n\nClosing Drone Mission Agent..."
            )
            break

        if not user_input:
            continue

        if user_input.lower() in {
            "exit",
            "quit",
            "salir",
        }:
            print(
                "\nClosing Drone Mission Agent..."
            )
            break

        if user_input.lower() == "reset":
            state = MissionState()
            last_analysis = None

            print(
                "\nMission reset."
            )

            continue

        if is_trace_request(
            user_input
        ):
            if last_analysis is None:
                print(
                    "\nAGENT:"
                    "\nNo mission has been analysed yet."
                )

            else:
                print(
                    "\nAGENT:"
                    "\nCalculation steps:"
                )

                print(
                    last_analysis["trace"]
                )

            continue

        updated_fields = update_state(
            state,
            user_input,
        )

        if (
            updated_fields
            or is_analysis_request(user_input)
        ):
            print(
                "\n[AGENT] Analysing mission..."
            )

            last_analysis = analyse_mission(
                state
            )

            print(
                "\nAGENT:"
            )

            print(
                last_analysis["report"]
            )

            continue

        print(
            "\n[AGENT] Thinking..."
        )

        try:
            answer = answer_question(
                user_input,
                state,
                last_analysis,
            )

        except Exception as error:
            print(
                "\n[ERROR]"
            )

            print(
                error
            )

            continue

        print(
            "\nAGENT:"
        )

        print(
            answer
        )


if __name__ == "__main__":
    main()