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
    updated_fields: list[str]

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

    run_flight: bool
    run_wind: bool
    run_battery: bool

    mission_status: str
    missing_fields: list[str]

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


FLIGHT_FIELDS = {
    "distance_km",
    "drone_speed_kmh",
    "wind_speed_kmh",
    "flight_bearing_deg",
    "wind_direction_from_deg",
}

WIND_FIELDS = {
    "wind_speed_kmh",
    "max_safe_wind_speed_kmh",
}

BATTERY_FIELDS = {
    "battery_percent",
    "consumption_percent_per_minute",
    "reserve_percent",
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
        wind_direction_from_deg
        - flight_bearing_deg
    )

    headwind_component_kmh = (
        wind_speed_kmh
        * math.cos(relative_angle_rad)
    )

    crosswind_component_kmh = (
        wind_speed_kmh
        * math.sin(relative_angle_rad)
    )

    if (
        abs(crosswind_component_kmh)
        >= drone_speed_kmh
    ):
        return {
            "status": "unsafe",
            "message": (
                "The drone cannot compensate for the crosswind."
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
        distance_km
        / ground_speed_kmh
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
        f"max_safe_wind_speed_kmh="
        f"{max_safe_wind_speed_kmh})"
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
        wind_speed_kmh
        <= max_safe_wind_speed_kmh
    )

    return {
        "status":
            "safe"
            if wind_safe
            else "unsafe",
        "wind_safe":
            wind_safe,
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
            "message": (
                "Flight time must be greater than 0."
            ),
        }

    if not 0 <= battery_percent <= 100:
        return {
            "status": "error",
            "message": (
                "Battery percentage must be between 0 and 100."
            ),
        }

    if consumption_percent_per_minute <= 0:
        return {
            "status": "error",
            "message": (
                "Battery consumption per minute "
                "must be greater than 0."
            ),
        }

    if not 0 <= reserve_percent <= 100:
        return {
            "status": "error",
            "message": (
                "Reserve percentage must be between 0 and 100."
            ),
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
        "status":
            "safe"
            if battery_sufficient
            else "unsafe",
        "battery_sufficient":
            battery_sufficient,
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


def to_float(
    value: str,
) -> float:
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

    return DIRECTIONS.get(
        direction
    )


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
        updates["distance_km"] = (
            to_float(
                match.group(1)
            )
        )

    speed_patterns = [
        (
            r"(?:travelling|traveling|flying|moving)"
            r"\s+at\s+"
            r"(\d+(?:[.,]\d+)?)"
            r"\s*km\s*/?\s*h"
        ),
        (
            r"drone(?:\s+speed|\s+airspeed)"
            r"[^0-9]{0,20}"
            r"(\d+(?:[.,]\d+)?)"
            r"\s*km\s*/?\s*h"
        ),
    ]

    for pattern in speed_patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            updates[
                "drone_speed_kmh"
            ] = to_float(
                match.group(1)
            )

            break

    max_wind_match = re.search(
        r"(?:maximum|max)\s+safe\s+wind\s+speed"
        r"(?:\s+for\s+the\s+drone)?"
        r"\s*(?:is|=|of|to|at)?\s*"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*km\s*/?\s*h",
        text,
        re.IGNORECASE,
    )

    wind_text = text

    if max_wind_match:
        updates[
            "max_safe_wind_speed_kmh"
        ] = to_float(
            max_wind_match.group(1)
        )

        start, end = (
            max_wind_match.span()
        )

        wind_text = (
            text[:start]
            + " " * (end - start)
            + text[end:]
        )

    match = re.search(
        r"\bwind\b(?:\s+speed)?"
        r"\s+(?:is|at|of|to)?\s*"
        r"(?:now\s+)?"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*km\s*/?\s*h",
        wind_text,
        re.IGNORECASE,
    )

    if match:
        updates[
            "wind_speed_kmh"
        ] = to_float(
            match.group(1)
        )

    battery_patterns = [
        (
            r"(?:the\s+)?drone\s+"
            r"(?:currently\s+)?has\s+"
            r"(\d+(?:[.,]\d+)?)\s*%"
            r"\s+battery"
            r"(?:\s+(?:remaining|left|available))?"
        ),
        (
            r"(\d+(?:[.,]\d+)?)\s*%"
            r"\s+battery\s+"
            r"(?:remaining|left|available)"
        ),
        (
            r"(?:current\s+)?battery"
            r"(?:\s+(?:level|percentage))?"
            r"\s*(?:is|at|=)\s*"
            r"(\d+(?:[.,]\d+)?)\s*%"
        ),
        (
            r"(?:set|change|update)\s+"
            r"(?:the\s+)?battery"
            r"(?:\s+(?:level|percentage))?"
            r"\s+(?:to|at)\s+"
            r"(\d+(?:[.,]\d+)?)\s*%"
        ),
    ]

    for pattern in battery_patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            updates[
                "battery_percent"
            ] = to_float(
                match.group(1)
            )

            break

    consumption_patterns = [
        (
            r"(?:consumes?|consumption"
            r"(?:\s+rate)?(?:\s+is)?)"
            r"[^0-9]{0,30}"
            r"(\d+(?:[.,]\d+)?)\s*%"
            r"(?:\s+battery)?"
            r"\s*(?:per\s+minute|/\s*min)"
        ),
        (
            r"(\d+(?:[.,]\d+)?)\s*%"
            r"(?:\s+battery)?"
            r"\s*(?:per\s+minute|/\s*min)"
        ),
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
        (
            r"(\d+(?:[.,]\d+)?)\s*%"
            r"(?:\s+battery)?"
            r"\s+reserve"
        ),
        (
            r"(?:reserve|minimum\s+reserve)"
            r"[^0-9]{0,30}"
            r"(\d+(?:[.,]\d+)?)\s*%"
        ),
    ]

    for pattern in reserve_patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            updates[
                "reserve_percent"
            ] = to_float(
                match.group(1)
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
        updates[
            "flight_bearing_deg"
        ] = (
            to_float(
                bearing_match.group(1)
            )
            % 360
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
                r"\b(?:towards?|heading|flight\s+direction|direction)"
                r"\s+(?:(?:to|is|=)\s+)?"
                r"(north(?:east|west)?|"
                r"south(?:east|west)?|"
                r"east|west|ne|nw|se|sw|n|s|e|w)\b",
            )

        if direction is not None:
            updates[
                "flight_bearing_deg"
            ] = direction

    wind_bearing_match = re.search(
        r"\bwind\b.*?\bfrom\s+(?:the\s+)?"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*(?:degrees|degree|deg|°)",
        text,
        re.IGNORECASE,
    )

    if wind_bearing_match:
        updates[
            "wind_direction_from_deg"
        ] = (
            to_float(
                wind_bearing_match.group(1)
            )
            % 360
        )

    else:
        wind_direction = extract_direction(
            text,
            r"\bwind\b.*?\bfrom\s+(?:the\s+)?"
            r"(north(?:east|west)?|"
            r"south(?:east|west)?|"
            r"east|west|ne|nw|se|sw|n|s|e|w)\b",
        )

        if wind_direction is None:
            wind_direction = extract_direction(
                text,
                r"\bwind\s+direction\b"
                r"\s+(?:(?:to|is|=)\s+)?"
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


def is_recalculation_request(
    user_input: str,
) -> bool:
    text = user_input.lower()

    return (
        "recalculate" in text
        or "recompute" in text
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
        f"{state.get('max_safe_wind_speed_kmh')}\n"
        f"mission_status={state.get('mission_status')}"
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


def get_changed_fields(
    state: DroneMissionState,
    updates: dict,
) -> list[str]:
    return [
        key
        for key, value in updates.items()
        if state.get(key) != value
    ]


def has_required_values(
    state: DroneMissionState,
    updates: dict,
    fields: set[str],
) -> bool:
    return all(
        updates.get(
            field,
            state.get(field),
        )
        is not None
        for field in fields
    )


def get_missing_fields(
    state: DroneMissionState,
) -> list[str]:
    field_labels = {
        "distance_km":
            "flight distance",
        "drone_speed_kmh":
            "drone speed",
        "wind_speed_kmh":
            "wind speed",
        "flight_bearing_deg":
            "flight direction or bearing",
        "wind_direction_from_deg":
            "wind direction",
        "max_safe_wind_speed_kmh":
            "maximum safe wind speed",
        "battery_percent":
            "current battery percentage",
        "consumption_percent_per_minute":
            "battery consumption percentage per minute",
        "reserve_percent":
            "required battery reserve",
    }

    return [
        label
        for field, label
        in field_labels.items()
        if state.get(field) is None
    ]


def build_calculation_trace(
    state: DroneMissionState,
) -> str:
    steps = [
        "Read the current mission parameters."
    ]

    flight_result = state.get(
        "flight_result"
    )

    wind_result = state.get(
        "wind_result"
    )

    battery_result = state.get(
        "battery_result"
    )

    if flight_result is None:
        steps.append(
            "Flight dynamics cannot be fully evaluated because "
            "a valid flight result is not available."
        )

    else:
        steps.append(
            "All parameters required for the flight dynamics "
            "analysis are available."
        )

        relative_angle = (
            state["wind_direction_from_deg"]
            - state["flight_bearing_deg"]
        )

        steps.append(
            "Calculate the relative wind angle: "
            f"{state['wind_direction_from_deg']}° - "
            f"{state['flight_bearing_deg']}° = "
            f"{relative_angle}°."
        )

        if (
            "headwind_component_kmh"
            in flight_result
        ):
            steps.append(
                "Headwind component: "
                f"{flight_result['headwind_component_kmh']} km/h."
            )

        if (
            "crosswind_component_kmh"
            in flight_result
        ):
            steps.append(
                "Crosswind component: "
                f"{flight_result['crosswind_component_kmh']} km/h."
            )

        if (
            flight_result["status"]
            == "success"
        ):
            steps.append(
                "The drone can compensate for the crosswind."
            )

            steps.append(
                "Ground speed after wind correction: "
                f"{flight_result['ground_speed_kmh']} km/h."
            )

            steps.append(
                "Calculate flight time: "
                f"{state['distance_km']} km / "
                f"{flight_result['ground_speed_kmh']} km/h × 60 = "
                f"{flight_result['flight_time_minutes']} minutes."
            )

        elif (
            flight_result["status"]
            == "unsafe"
        ):
            if (
                "crosswind_component_kmh"
                in flight_result
                and state.get(
                    "drone_speed_kmh"
                )
                is not None
                and abs(
                    flight_result[
                        "crosswind_component_kmh"
                    ]
                )
                >= state[
                    "drone_speed_kmh"
                ]
            ):
                steps.append(
                    "Compare the crosswind magnitude with the "
                    "drone airspeed: "
                    f"{abs(flight_result['crosswind_component_kmh'])} "
                    f"km/h >= {state['drone_speed_kmh']} km/h."
                )

            steps.append(
                flight_result["message"]
            )

        else:
            steps.append(
                flight_result.get(
                    "message",
                    (
                        "The flight dynamics calculation "
                        "returned an error."
                    ),
                )
            )

    if wind_result is None:
        steps.append(
            "Wind safety cannot be fully evaluated because "
            "a valid wind safety result is not available."
        )

    else:
        steps.append(
            "Compare the wind speed with the drone safety limit: "
            f"{state['wind_speed_kmh']} km/h versus "
            f"{state['max_safe_wind_speed_kmh']} km/h."
        )

        if (
            wind_result["status"]
            == "safe"
        ):
            steps.append(
                "The wind speed is within the drone's general "
                "safe operating limit."
            )

        elif (
            wind_result["status"]
            == "unsafe"
        ):
            steps.append(
                "The wind speed exceeds the drone's general "
                "safe operating limit."
            )

        else:
            steps.append(
                wind_result.get(
                    "message",
                    "The wind safety check returned an error.",
                )
            )

    if battery_result is not None:
        steps.append(
            "Calculate required battery: "
            f"{flight_result['flight_time_minutes']} minutes × "
            f"{state['consumption_percent_per_minute']}%/min = "
            f"{battery_result['required_battery_percent']}%."
        )

        steps.append(
            "Calculate battery after flight: "
            f"{state['battery_percent']}% - "
            f"{battery_result['required_battery_percent']}% = "
            f"{battery_result['battery_after_flight_percent']}%."
        )

        steps.append(
            "Compare the remaining battery with the required "
            f"{state['reserve_percent']}% reserve."
        )

        if (
            battery_result["status"]
            == "safe"
        ):
            steps.append(
                "The remaining battery satisfies "
                "the required reserve."
            )

        elif (
            battery_result["status"]
            == "unsafe"
        ):
            steps.append(
                "The remaining battery does not satisfy "
                "the required reserve."
            )

        else:
            steps.append(
                battery_result.get(
                    "message",
                    "The battery safety check returned an error.",
                )
            )

    elif (
        flight_result is not None
        and flight_result.get("status")
        != "success"
    ):
        steps.append(
            "Battery analysis is not performed because the "
            "flight dynamics do not provide a valid flight time."
        )

    else:
        battery_data_available = all(
            state.get(field) is not None
            for field in BATTERY_FIELDS
        )

        if not battery_data_available:
            steps.append(
                "Battery safety cannot yet be fully calculated "
                "because one or more battery parameters are missing."
            )

        else:
            steps.append(
                "Battery safety does not currently have "
                "a valid result."
            )

    status = state.get(
        "mission_status",
        "UNDETERMINED",
    )

    if status == "FEASIBLE":
        steps.append(
            "Flight dynamics, wind safety and battery safety "
            "all passed. The mission is FEASIBLE."
        )

    elif status == "NOT FEASIBLE":
        steps.append(
            "At least one safety condition failed, therefore "
            "the mission is NOT FEASIBLE."
        )

    elif status == "INVALID":
        steps.append(
            "At least one validation check returned an error, "
            "therefore the mission data is INVALID."
        )

    else:
        steps.append(
            "Mission feasibility remains UNDETERMINED because "
            "required information or results are still missing."
        )

    return "\n".join(
        f"{number}. {step}"
        for number, step
        in enumerate(
            steps,
            start=1,
        )
    )


def process_input_node(
    state: DroneMissionState,
) -> dict:
    print(
        "\n[NODE] process_input"
    )

    user_input = state[
        "user_input"
    ]

    if is_trace_request(
        user_input
    ):
        return {
            "intent": "trace",
            "updated_fields": [],
        }

    updates = extract_updates(
        user_input
    )

    if (
        not updates
        and not is_analysis_request(
            user_input
        )
    ):
        return {
            "intent": "question",
            "updated_fields": [],
        }

    changed_fields = (
        get_changed_fields(
            state,
            updates,
        )
    )

    changed_set = set(
        changed_fields
    )

    force_recalculation = (
        is_recalculation_request(
            user_input
        )
    )

    flight_changed = bool(
        changed_set
        & FLIGHT_FIELDS
    )

    wind_changed = bool(
        changed_set
        & WIND_FIELDS
    )

    battery_changed = bool(
        changed_set
        & BATTERY_FIELDS
    )

    flight_result = state.get(
        "flight_result"
    )

    wind_result = state.get(
        "wind_result"
    )

    battery_result = state.get(
        "battery_result"
    )

    if (
        flight_changed
        or force_recalculation
    ):
        flight_result = None
        battery_result = None

    if (
        wind_changed
        or force_recalculation
    ):
        wind_result = None

    if (
        battery_changed
        or force_recalculation
    ):
        battery_result = None

    flight_ready = (
        has_required_values(
            state,
            updates,
            FLIGHT_FIELDS,
        )
    )

    wind_ready = (
        has_required_values(
            state,
            updates,
            WIND_FIELDS,
        )
    )

    battery_ready = (
        has_required_values(
            state,
            updates,
            BATTERY_FIELDS,
        )
    )

    run_flight = (
        flight_ready
        and flight_result is None
    )

    run_wind = (
        wind_ready
        and wind_result is None
    )

    run_battery = (
        battery_ready
        and battery_result is None
    )

    scheduled_checks = []

    if run_flight:
        scheduled_checks.append(
            "flight"
        )

    if run_wind:
        scheduled_checks.append(
            "wind"
        )

    if run_battery:
        scheduled_checks.append(
            "battery"
        )

    if changed_fields:
        print(
            "[STATE] Updated fields: "
            + ", ".join(
                changed_fields
            )
        )

    else:
        print(
            "[STATE] No mission parameter changed."
        )

    if scheduled_checks:
        print(
            "[ROUTER] Scheduled checks: "
            + " -> ".join(
                scheduled_checks
            )
        )

    else:
        print(
            "[ROUTER] Reusing cached results."
        )

    return {
        **updates,
        "intent":
            "analyse",
        "updated_fields":
            changed_fields,
        "flight_result":
            flight_result,
        "wind_result":
            wind_result,
        "battery_result":
            battery_result,
        "run_flight":
            run_flight,
        "run_wind":
            run_wind,
        "run_battery":
            run_battery,
        "response":
            "",
    }


def route_after_input(
    state: DroneMissionState,
) -> Literal[
    "flight",
    "wind",
    "battery",
    "report",
    "trace",
    "question",
]:
    intent = state[
        "intent"
    ]

    if intent == "trace":
        return "trace"

    if intent == "question":
        return "question"

    if state.get(
        "run_flight",
        False,
    ):
        return "flight"

    if state.get(
        "run_wind",
        False,
    ):
        return "wind"

    if (
        state.get(
            "run_battery",
            False,
        )
        and state.get(
            "flight_result"
        )
        is not None
        and state[
            "flight_result"
        ].get("status")
        == "success"
    ):
        return "battery"

    return "report"


def flight_check_node(
    state: DroneMissionState,
) -> dict:
    print(
        "[NODE] flight_check"
    )

    flight_result = (
        calculate_flight_time.invoke(
            {
                "distance_km":
                    state[
                        "distance_km"
                    ],
                "drone_speed_kmh":
                    state[
                        "drone_speed_kmh"
                    ],
                "wind_speed_kmh":
                    state[
                        "wind_speed_kmh"
                    ],
                "flight_bearing_deg":
                    state[
                        "flight_bearing_deg"
                    ],
                "wind_direction_from_deg":
                    state[
                        "wind_direction_from_deg"
                    ],
            }
        )
    )

    return {
        "flight_result":
            flight_result,
        "run_flight":
            False,
        "run_battery":
            (
                state.get(
                    "run_battery",
                    False,
                )
                and flight_result.get(
                    "status"
                )
                == "success"
            ),
    }


def route_after_flight(
    state: DroneMissionState,
) -> Literal[
    "wind",
    "battery",
    "report",
]:
    if state.get(
        "run_wind",
        False,
    ):
        return "wind"

    if (
        state.get(
            "run_battery",
            False,
        )
        and state.get(
            "flight_result"
        )
        is not None
        and state[
            "flight_result"
        ].get("status")
        == "success"
    ):
        return "battery"

    return "report"


def wind_check_node(
    state: DroneMissionState,
) -> dict:
    print(
        "[NODE] wind_check"
    )

    wind_result = (
        check_wind_safety.invoke(
            {
                "wind_speed_kmh":
                    state[
                        "wind_speed_kmh"
                    ],
                "max_safe_wind_speed_kmh":
                    state[
                        "max_safe_wind_speed_kmh"
                    ],
            }
        )
    )

    return {
        "wind_result":
            wind_result,
        "run_wind":
            False,
    }


def route_after_wind(
    state: DroneMissionState,
) -> Literal[
    "battery",
    "report",
]:
    if (
        state.get(
            "run_battery",
            False,
        )
        and state.get(
            "flight_result"
        )
        is not None
        and state[
            "flight_result"
        ].get("status")
        == "success"
    ):
        return "battery"

    return "report"


def battery_check_node(
    state: DroneMissionState,
) -> dict:
    print(
        "[NODE] battery_check"
    )

    flight_result = state[
        "flight_result"
    ]

    battery_result = (
        check_battery.invoke(
            {
                "flight_time_minutes":
                    flight_result[
                        "flight_time_minutes"
                    ],
                "battery_percent":
                    state[
                        "battery_percent"
                    ],
                "consumption_percent_per_minute":
                    state[
                        "consumption_percent_per_minute"
                    ],
                "reserve_percent":
                    state[
                        "reserve_percent"
                    ],
            }
        )
    )

    return {
        "battery_result":
            battery_result,
        "run_battery":
            False,
    }


def final_report_node(
    state: DroneMissionState,
) -> dict:
    print(
        "[NODE] final_report"
    )

    flight_result = state.get(
        "flight_result"
    )

    wind_result = state.get(
        "wind_result"
    )

    battery_result = state.get(
        "battery_result"
    )

    output = []

    unsafe = False
    invalid = False

    if flight_result is None:
        output.append(
            "Flight calculation: UNDETERMINED"
        )

    elif (
        flight_result["status"]
        == "success"
    ):
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

    elif (
        flight_result["status"]
        == "unsafe"
    ):
        unsafe = True

        output.append(
            "Flight dynamics: UNSAFE"
        )

        output.append(
            flight_result[
                "message"
            ]
        )

    else:
        invalid = True

        output.append(
            "Flight calculation: ERROR"
        )

        output.append(
            flight_result.get(
                "message",
                (
                    "Unknown flight "
                    "calculation error."
                ),
            )
        )

    if wind_result is None:
        output.append(
            "Wind safety: UNDETERMINED"
        )

    elif (
        wind_result["status"]
        == "safe"
    ):
        output.append(
            "Wind safety: SAFE "
            f"({wind_result['wind_speed_kmh']} km/h <= "
            f"{wind_result['max_safe_wind_speed_kmh']} km/h)"
        )

    elif (
        wind_result["status"]
        == "unsafe"
    ):
        unsafe = True

        output.append(
            "Wind safety: UNSAFE "
            f"({wind_result['wind_speed_kmh']} km/h > "
            f"{wind_result['max_safe_wind_speed_kmh']} km/h)"
        )

    else:
        invalid = True

        output.append(
            "Wind safety: ERROR"
        )

        output.append(
            wind_result.get(
                "message",
                (
                    "Unknown wind "
                    "safety error."
                ),
            )
        )

    if battery_result is None:
        output.append(
            "Battery safety: UNDETERMINED"
        )

    elif (
        battery_result["status"]
        == "safe"
    ):
        output.append(
            "Battery safety: SAFE"
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

    elif (
        battery_result["status"]
        == "unsafe"
    ):
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

    else:
        invalid = True

        output.append(
            "Battery safety: ERROR"
        )

        output.append(
            battery_result.get(
                "message",
                (
                    "Unknown battery "
                    "safety error."
                ),
            )
        )

    missing_fields = (
        get_missing_fields(
            state
        )
    )

    if invalid:
        mission_status = (
            "INVALID"
        )

        output.append(
            "\nFinal mission feasibility: INVALID DATA"
        )

    elif unsafe:
        mission_status = (
            "NOT FEASIBLE"
        )

        output.append(
            "\nFinal mission feasibility: NOT FEASIBLE"
        )

    elif missing_fields:
        mission_status = (
            "UNDETERMINED"
        )

        output.append(
            "\nMissing information:"
        )

        for item in missing_fields:
            output.append(
                f"- {item}"
            )

        output.append(
            "\nFinal mission feasibility: UNDETERMINED"
        )

    elif (
        flight_result is None
        or wind_result is None
        or battery_result is None
    ):
        mission_status = (
            "UNDETERMINED"
        )

        output.append(
            "\nFinal mission feasibility: UNDETERMINED"
        )

    else:
        mission_status = (
            "FEASIBLE"
        )

        output.append(
            "\nFinal mission feasibility: FEASIBLE"
        )

    report = "\n".join(
        output
    )

    temporary_state = {
        **state,
        "mission_status":
            mission_status,
        "missing_fields":
            missing_fields,
    }

    calculation_trace = (
        build_calculation_trace(
            temporary_state
        )
    )

    return {
        "mission_status":
            mission_status,
        "missing_fields":
            missing_fields,
        "final_report":
            report,
        "calculation_trace":
            calculation_trace,
        "response":
            report,
    }


def show_trace_node(
    state: DroneMissionState,
) -> dict:
    print(
        "[NODE] show_trace"
    )

    calculation_trace = (
        state.get(
            "calculation_trace"
        )
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
        "response":
            response,
    }


def answer_question_node(
    state: DroneMissionState,
) -> dict:
    print(
        "[NODE] answer_question"
    )

    print(
        "[AGENT] Thinking..."
    )

    report = state.get(
        "final_report",
        (
            "No mission has "
            "been analysed yet."
        ),
    )

    trace = state.get(
        "calculation_trace",
        (
            "No calculation steps "
            "are available yet."
        ),
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

The magnitude represents crosswind strength.
The sign represents direction according to this mathematical convention.

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
                state[
                    "user_input"
                ],
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
        "flight":
            "flight_check",
        "wind":
            "wind_check",
        "battery":
            "battery_check",
        "report":
            "final_report",
        "trace":
            "show_trace",
        "question":
            "answer_question",
    },
)


builder.add_conditional_edges(
    "flight_check",
    route_after_flight,
    {
        "wind":
            "wind_check",
        "battery":
            "battery_check",
        "report":
            "final_report",
    },
)


builder.add_conditional_edges(
    "wind_check",
    route_after_wind,
    {
        "battery":
            "battery_check",
        "report":
            "final_report",
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


drone_mission_graph = (
    builder.compile(
        checkpointer=memory
    )
)


THREAD_ID = (
    "drone-mission-session"
)


config = {
    "configurable": {
        "thread_id":
            THREAD_ID
    }
}


def main() -> None:
    print(
        "LangGraph Drone Mission Agent"
    )

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

        if (
            user_input.lower()
            == "reset"
        ):
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
            result = (
                drone_mission_graph.invoke(
                    {
                        "user_input":
                            user_input
                    },
                    config=config,
                )
            )

        except Exception as error:  # noqa: BLE001 - Disable Ruff
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
            result[
                "response"
            ]
        )


if __name__ == "__main__":
    main()