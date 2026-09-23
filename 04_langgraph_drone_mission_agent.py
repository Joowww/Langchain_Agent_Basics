import math
import re
from typing import Literal, TypedDict

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class DroneMissionState(TypedDict, total=False):
    user_input: str
    intent: str
    response: str

    distance_km: float | None
    drone_speed_kmh: float | None
    wind_speed_kmh: float | None
    flight_bearing_deg: float | None
    wind_direction_from_deg: float | None
    battery_percent: float | None
    consumption_percent_per_minute: float | None
    reserve_percent: float | None
    max_safe_wind_speed_kmh: float | None

    flight_result: dict | None
    wind_result: dict | None
    battery_result: dict | None

    trace_steps: list[str]
    final_report: str
    calculation_trace: str


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


def extract_updates(
    user_input: str,
) -> dict:
    text = user_input.lower()
    updates = {}

    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*"
        r"(?:km|kilometres?|kilometers?)"
        r"(?!\s*/?\s*h)",
        text,
        re.IGNORECASE,
    )

    if match:
        updates["distance_km"] = to_float(
            match.group(1)
        )

    speed_patterns = [
        r"(?:travelling|traveling|flying|moving)"   # noqa: ISC004 - Comment to disable Ruff for these lines
        r"\s+at\s+"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*km\s*/?\s*h",

        r"drone(?:\s+speed|\s+airspeed)"   # noqa: ISC004 - Comment to disable Ruff for these lines
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
            updates["drone_speed_kmh"] = (
                to_float(match.group(1))
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
        updates["wind_speed_kmh"] = to_float(
            match.group(1)
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
        updates["max_safe_wind_speed_kmh"] = (
            to_float(match.group(1))
        )

    battery_patterns = [
        r"(?:the\s+)?drone\s+"    # noqa: ISC004 - Comment to disable Ruff for these lines
        r"(?:currently\s+)?has\s+"
        r"(\d+(?:[.,]\d+)?)\s*%"
        r"\s+battery"
        r"(?:\s+(?:remaining|left|available))?",

        r"(\d+(?:[.,]\d+)?)\s*%"   # noqa: ISC004 - Comment to disable Ruff for these lines
        r"\s+battery\s+"
        r"(?:remaining|left|available)",

        r"(?:current\s+)?battery"      # noqa: ISC004 - Comment to disable Ruff for these lines
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
            updates["battery_percent"] = (
                to_float(match.group(1))
            )
            break

    consumption_patterns = [
        r"(?:consumes?|consumption"      # noqa: ISC004 - Comment to disable Ruff for these lines
        r"(?:\s+rate)?(?:\s+is)?)"
        r"[^0-9]{0,30}"
        r"(\d+(?:[.,]\d+)?)\s*%"
        r"(?:\s+battery)?"
        r"\s*(?:per\s+minute|/\s*min)",

        r"(\d+(?:[.,]\d+)?)\s*%"      # noqa: ISC004 - Comment to disable Ruff for these lines
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
            updates[
                "consumption_percent_per_minute"
            ] = to_float(
                match.group(1)
            )
            break

    reserve_patterns = [
        r"(\d+(?:[.,]\d+)?)\s*%"      # noqa: ISC004 - Comment to disable Ruff for these lines
        r"(?:\s+battery)?"
        r"\s+reserve",

        r"(?:reserve|minimum\s+reserve)"      # noqa: ISC004 - Comment to disable Ruff for these lines
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
            updates["reserve_percent"] = (
                to_float(match.group(1))
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
        updates["flight_bearing_deg"] = (
            to_float(
                bearing_match.group(1)
            ) % 360
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
            updates["flight_bearing_deg"] = direction

    wind_bearing_match = re.search(
        r"wind.*?\bfrom\s+(?:the\s+)?"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*(?:degrees|degree|deg|°)",
        text,
        re.IGNORECASE,
    )

    if wind_bearing_match:
        updates["wind_direction_from_deg"] = (
            to_float(
                wind_bearing_match.group(1)
            ) % 360
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
            updates[
                "wind_direction_from_deg"
            ] = wind_direction

    return updates


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


def state_to_text(
    state: DroneMissionState,
) -> str:
    return (
        f"distance_km={state.get('distance_km')}\n"
        f"drone_speed_kmh={state.get('drone_speed_kmh')}\n"
        f"wind_speed_kmh={state.get('wind_speed_kmh')}\n"
        f"flight_bearing_deg={state.get('flight_bearing_deg')}\n"
        f"wind_direction_from_deg="
        f"{state.get('wind_direction_from_deg')}\n"
        f"battery_percent={state.get('battery_percent')}\n"
        f"consumption_percent_per_minute="
        f"{state.get('consumption_percent_per_minute')}\n"
        f"reserve_percent={state.get('reserve_percent')}\n"
        f"max_safe_wind_speed_kmh="
        f"{state.get('max_safe_wind_speed_kmh')}"
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


def process_input_node(
    state: DroneMissionState,
) -> dict:
    print("\n[NODE] process_input")

    user_input = state["user_input"]

    if is_trace_request(
        user_input
    ):
        return {
            "intent": "trace",
        }

    updates = extract_updates(
        user_input
    )

    if (
        updates
        or is_analysis_request(user_input)
    ):
        return {
            **updates,
            "intent": "analyse",
            "flight_result": None,
            "wind_result": None,
            "battery_result": None,
            "trace_steps": [
                "Read the current mission parameters."
            ],
            "response": "",
        }

    return {
        "intent": "question",
    }


def route_after_input(
    state: DroneMissionState,
) -> Literal[
    "analyse",
    "trace",
    "question",
]:
    return state["intent"]


def flight_check_node(
    state: DroneMissionState,
) -> dict:
    print("[NODE] flight_check")

    trace = list(
        state.get(
            "trace_steps",
            [],
        )
    )

    required_values = [
        state.get("distance_km"),
        state.get("drone_speed_kmh"),
        state.get("wind_speed_kmh"),
        state.get("flight_bearing_deg"),
        state.get("wind_direction_from_deg"),
    ]

    if not all(
        value is not None
        for value in required_values
    ):
        trace.append(
            "Flight dynamics cannot be calculated because one "
            "or more required parameters are missing."
        )

        return {
            "flight_result": None,
            "trace_steps": trace,
        }

    trace.append(
        "All parameters required for the flight dynamics "
        "analysis are available."
    )

    flight_result = calculate_flight_time.invoke(
        {
            "distance_km":
                state["distance_km"],
            "drone_speed_kmh":
                state["drone_speed_kmh"],
            "wind_speed_kmh":
                state["wind_speed_kmh"],
            "flight_bearing_deg":
                state["flight_bearing_deg"],
            "wind_direction_from_deg":
                state["wind_direction_from_deg"],
        }
    )

    relative_angle = (
        state["wind_direction_from_deg"]
        - state["flight_bearing_deg"]
    )

    trace.append(
        "Calculate the relative wind angle: "
        f"{state['wind_direction_from_deg']}° - "
        f"{state['flight_bearing_deg']}° = "
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
            f"{state['distance_km']} km / "
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
            >= state["drone_speed_kmh"]
        ):
            trace.append(
                "Compare the crosswind magnitude with the "
                "drone airspeed: "
                f"{abs(flight_result['crosswind_component_kmh'])} "
                f"km/h >= {state['drone_speed_kmh']} km/h."
            )

        trace.append(
            flight_result["message"]
        )

    return {
        "flight_result": flight_result,
        "trace_steps": trace,
    }


def wind_check_node(
    state: DroneMissionState,
) -> dict:
    print("[NODE] wind_check")

    trace = list(
        state.get(
            "trace_steps",
            [],
        )
    )

    if (
        state.get("wind_speed_kmh") is None
        or state.get(
            "max_safe_wind_speed_kmh"
        ) is None
    ):
        trace.append(
            "Wind safety cannot be determined because required "
            "information is missing."
        )

        return {
            "wind_result": None,
            "trace_steps": trace,
        }

    wind_result = check_wind_safety.invoke(
        {
            "wind_speed_kmh":
                state["wind_speed_kmh"],
            "max_safe_wind_speed_kmh":
                state[
                    "max_safe_wind_speed_kmh"
                ],
        }
    )

    trace.append(
        "Compare the wind speed with the drone safety limit: "
        f"{state['wind_speed_kmh']} km/h versus "
        f"{state['max_safe_wind_speed_kmh']} km/h."
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

    return {
        "wind_result": wind_result,
        "trace_steps": trace,
    }


def route_after_wind(
    state: DroneMissionState,
) -> Literal[
    "battery",
    "report",
]:
    flight_result = state.get(
        "flight_result"
    )

    battery_data_available = all(
        state.get(key) is not None
        for key in [
            "battery_percent",
            "consumption_percent_per_minute",
            "reserve_percent",
        ]
    )

    if (
        flight_result is not None
        and flight_result.get("status")
        == "success"
        and battery_data_available
    ):
        return "battery"

    return "report"


def battery_check_node(
    state: DroneMissionState,
) -> dict:
    print("[NODE] battery_check")

    trace = list(
        state.get(
            "trace_steps",
            [],
        )
    )

    flight_result = state["flight_result"]

    battery_result = check_battery.invoke(
        {
            "flight_time_minutes":
                flight_result[
                    "flight_time_minutes"
                ],
            "battery_percent":
                state["battery_percent"],
            "consumption_percent_per_minute":
                state[
                    "consumption_percent_per_minute"
                ],
            "reserve_percent":
                state["reserve_percent"],
        }
    )

    trace.append(
        "Calculate required battery: "
        f"{flight_result['flight_time_minutes']} minutes × "
        f"{state['consumption_percent_per_minute']}%/min = "
        f"{battery_result['required_battery_percent']}%."
    )

    trace.append(
        "Calculate battery after flight: "
        f"{state['battery_percent']}% - "
        f"{battery_result['required_battery_percent']}% = "
        f"{battery_result['battery_after_flight_percent']}%."
    )

    trace.append(
        "Compare the remaining battery with the required "
        f"{state['reserve_percent']}% reserve."
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

    return {
        "battery_result": battery_result,
        "trace_steps": trace,
    }


def final_report_node(
    state: DroneMissionState,
) -> dict:
    print("[NODE] final_report")

    flight_result = state.get(
        "flight_result"
    )

    wind_result = state.get(
        "wind_result"
    )

    battery_result = state.get(
        "battery_result"
    )

    trace = list(
        state.get(
            "trace_steps",
            [],
        )
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

        battery_parameters_available = all(
            state.get(key) is not None
            for key in [
                "battery_percent",
                "consumption_percent_per_minute",
                "reserve_percent",
            ]
        )

        if (
            flight_result is not None
            and flight_result.get("status")
            != "success"
        ):
            trace.append(
                "Battery analysis is not performed because the "
                "flight dynamics are already unsafe and no valid "
                "flight time is available."
            )

        elif not battery_parameters_available:
            trace.append(
                "Battery safety cannot yet be fully calculated "
                "because one or more battery parameters are missing."
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

    else:
        missing = []

        if state.get("distance_km") is None:
            missing.append(
                "flight distance"
            )

        if state.get("drone_speed_kmh") is None:
            missing.append(
                "drone speed"
            )

        if state.get("wind_speed_kmh") is None:
            missing.append(
                "wind speed"
            )

        if state.get("flight_bearing_deg") is None:
            missing.append(
                "flight direction or bearing"
            )

        if state.get(
            "wind_direction_from_deg"
        ) is None:
            missing.append(
                "wind direction"
            )

        if state.get(
            "max_safe_wind_speed_kmh"
        ) is None:
            missing.append(
                "maximum safe wind speed"
            )

        if state.get("battery_percent") is None:
            missing.append(
                "current battery percentage"
            )

        if state.get(
            "consumption_percent_per_minute"
        ) is None:
            missing.append(
                "battery consumption percentage per minute"
            )

        if state.get("reserve_percent") is None:
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
                "Flight dynamics, wind safety and battery safety "
                "all passed. The mission is FEASIBLE."
            )

    report = "\n".join(
        output
    )

    calculation_trace = format_trace(
        trace
    )

    return {
        "final_report": report,
        "calculation_trace":
            calculation_trace,
        "trace_steps": trace,
        "response": report,
    }


def show_trace_node(
    state: DroneMissionState,
) -> dict:
    print("[NODE] show_trace")

    calculation_trace = state.get(
        "calculation_trace"
    )

    if not calculation_trace:
        response = (
            "No mission has been analysed yet."
        )

    else:
        response = (
            "Calculation steps:\n"
            + calculation_trace
        )

    return {
        "response": response,
    }


def answer_question_node(
    state: DroneMissionState,
) -> dict:
    print("[NODE] answer_question")
    print("[AGENT] Thinking...")

    report = state.get(
        "final_report",
        "No mission has been analysed yet.",
    )

    trace = state.get(
        "calculation_trace",
        "No calculation steps are available yet.",
    )

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
                state["user_input"],
            ),
        ],
        reasoning=False,
    )

    return {
        "response":
            clean_llm_response(
                response.content
            )
    }


builder = StateGraph(
    DroneMissionState
)

builder.add_node(
    "process_input",
    process_input_node,
)

builder.add_node(
    "flight_check",
    flight_check_node,
)

builder.add_node(
    "wind_check",
    wind_check_node,
)

builder.add_node(
    "battery_check",
    battery_check_node,
)

builder.add_node(
    "final_report",
    final_report_node,
)

builder.add_node(
    "show_trace",
    show_trace_node,
)

builder.add_node(
    "answer_question",
    answer_question_node,
)


builder.add_edge(
    START,
    "process_input",
)


builder.add_conditional_edges(
    "process_input",
    route_after_input,
    {
        "analyse": "flight_check",
        "trace": "show_trace",
        "question": "answer_question",
    },
)


builder.add_edge(
    "flight_check",
    "wind_check",
)


builder.add_conditional_edges(
    "wind_check",
    route_after_wind,
    {
        "battery": "battery_check",
        "report": "final_report",
    },
)


builder.add_edge(
    "battery_check",
    "final_report",
)


builder.add_edge(
    "final_report",
    END,
)

builder.add_edge(
    "show_trace",
    END,
)

builder.add_edge(
    "answer_question",
    END,
)


memory = InMemorySaver()

drone_mission_graph = builder.compile(
    checkpointer=memory
)


THREAD_ID = "drone-mission-session"

config = {
    "configurable": {
        "thread_id": THREAD_ID
    }
}


def main() -> None:
    print("LangGraph Drone Mission Agent")

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
                "\n\nClosing LangGraph Drone Mission Agent..."
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
                "\nClosing LangGraph Drone Mission Agent..."
            )
            break

        if user_input.lower() == "reset":
            memory.delete_thread(
                THREAD_ID
            )

            print(
                "\nMission reset."
            )

            continue

        print(
            "\n[LANGGRAPH] Running graph..."
        )

        try:
            result = drone_mission_graph.invoke(
                {
                    "user_input":
                        user_input
                },
                config=config,
            )

        except Exception as error:  # noqa: BLE001 Disbale exception Ruff
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
            result["response"]
        )


if __name__ == "__main__":
    main()