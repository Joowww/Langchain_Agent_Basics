import math
import re
from typing import Literal, TypedDict

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class AutonomousMissionState(TypedDict, total=False):
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
    failure_reasons: list[str]

    plan_version: int
    mission_plan: str

    replan_status: str
    replan_actions: list[str]
    replan_deferred_actions: list[str]
    replan_requires_reevaluation: bool

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
    """Calculate drone flight time considering wind."""

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
    """Check whether the wind speed is safe for the drone."""

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
    """Check whether the drone has enough battery for the flight."""

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
            "message": (
                "Battery percentage must be between 0 and 100."
            ),
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
        "status": (
            "safe"
            if battery_sufficient
            else "unsafe"
        ),
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
        updates["distance_km"] = to_float(
            match.group(1)
        )

    speed_patterns = [
        (
            r"(?:travelling|traveling|flying|moving)"
            r"\s+at\s+"
            r"(\d+(?:[.,]\d+)?)"
            r"\s*km\s*/?\s*h"
        ),
        (
            r"\bdrone\s+(?:speed|airspeed)\b"
            r"[^0-9]{0,20}"
            r"(\d+(?:[.,]\d+)?)"
            r"\s*km\s*/?\s*h"
        ),
        (
            r"(?:change|set|update)\s+"
            r"(?:the\s+)?drone\s+"
            r"(?:speed|airspeed)"
            r"\s+(?:to|at)\s+"
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

        start, end = max_wind_match.span()

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
            r"(?:current\s+)?battery"
            r"(?:\s+(?:level|percentage))?"
            r"\s*(?:is|at|=|to)\s*"
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
        direction_patterns = [
            (
                r"\b(?:fly|flying)\b.*?"
                r"\b(?:km|kilometres?|kilometers?)\b\s+"
                r"(north(?:east|west)?|"
                r"south(?:east|west)?|"
                r"east|west|ne|nw|se|sw|n|s|e|w)\b"
            ),
            (
                r"(?:change|set|update)\s+"
                r"(?:the\s+)?(?:flight\s+)?"
                r"(?:direction|heading)"
                r"\s+(?:to|at)\s+"
                r"(north(?:east|west)?|"
                r"south(?:east|west)?|"
                r"east|west|ne|nw|se|sw|n|s|e|w)\b"
            ),
        ]

        for pattern in direction_patterns:
            direction = extract_direction(
                text,
                pattern,
            )

            if direction is not None:
                updates[
                    "flight_bearing_deg"
                ] = direction
                break

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
        wind_direction_patterns = [
            (
                r"\bwind\b.*?\bfrom\s+(?:the\s+)?"
                r"(north(?:east|west)?|"
                r"south(?:east|west)?|"
                r"east|west|ne|nw|se|sw|n|s|e|w)\b"
            ),
            (
                r"(?:change|set|update)\s+"
                r"(?:the\s+)?wind\s+direction"
                r"\s+(?:to|at|from)\s+(?:the\s+)?"
                r"(north(?:east|west)?|"
                r"south(?:east|west)?|"
                r"east|west|ne|nw|se|sw|n|s|e|w)\b"
            ),
        ]

        for pattern in wind_direction_patterns:
            wind_direction = extract_direction(
                text,
                pattern,
            )

            if wind_direction is not None:
                updates[
                    "wind_direction_from_deg"
                ] = wind_direction
                break

    return updates


def is_trace_request(
    user_input: str,
) -> bool:
    text = user_input.lower()

    return any(
        phrase in text
        for phrase in [
            "all the steps",
            "show me the steps",
            "show the steps",
            "explain the steps",
            "how did you calculate",
            "how did you get",
            "how did you reach",
        ]
    )


def is_analysis_request(
    user_input: str,
) -> bool:
    text = user_input.lower()

    return any(
        phrase in text
        for phrase in [
            "is the mission feasible",
            "is this mission feasible",
            "check the mission",
            "analyse the mission",
            "analyze the mission",
            "recalculate",
            "recompute",
            "check feasibility",
            "plan the mission",
            "replan the mission",
        ]
    )


def is_recalculation_request(
    user_input: str,
) -> bool:
    text = user_input.lower()

    return (
        "recalculate" in text
        or "recompute" in text
        or "replan the mission" in text
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
        content = re.split(
            r"</think>",
            content,
            flags=re.IGNORECASE,
            maxsplit=1,
        )[-1]

    return content.strip()


def get_changed_fields(
    state: AutonomousMissionState,
    updates: dict,
) -> list[str]:
    return [
        key
        for key, value in updates.items()
        if state.get(key) != value
    ]


def has_required_values(
    state: AutonomousMissionState,
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
    state: AutonomousMissionState,
) -> list[str]:
    labels = {
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
        for field, label in labels.items()
        if state.get(field) is None
    ]


def bearing_name(
    bearing: float | None,
) -> str:
    names = {
        0.0: "north",
        45.0: "northeast",
        90.0: "east",
        135.0: "southeast",
        180.0: "south",
        225.0: "southwest",
        270.0: "west",
        315.0: "northwest",
    }

    if bearing is None:
        return "unknown"

    normalized = bearing % 360

    return names.get(
        normalized,
        f"{normalized} degrees",
    )


def state_to_text(
    state: AutonomousMissionState,
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
        f"mission_status={state.get('mission_status')}\n"
        f"replan_status={state.get('replan_status')}"
    )


def process_input_node(
    state: AutonomousMissionState,
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

    changed_fields = get_changed_fields(
        state,
        updates,
    )

    changed_set = set(
        changed_fields
    )

    force = is_recalculation_request(
        user_input
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
        changed_set & FLIGHT_FIELDS
        or force
    ):
        flight_result = None
        battery_result = None

    if (
        changed_set & WIND_FIELDS
        or force
    ):
        wind_result = None

    if (
        changed_set & BATTERY_FIELDS
        or force
    ):
        battery_result = None

    run_flight = (
        has_required_values(
            state,
            updates,
            FLIGHT_FIELDS,
        )
        and flight_result is None
    )

    run_wind = (
        has_required_values(
            state,
            updates,
            WIND_FIELDS,
        )
        and wind_result is None
    )

    run_battery = (
        has_required_values(
            state,
            updates,
            BATTERY_FIELDS,
        )
        and battery_result is None
    )

    scheduled = []

    if run_flight:
        scheduled.append("flight")

    if run_wind:
        scheduled.append("wind")

    if run_battery:
        scheduled.append("battery")

    if changed_fields:
        print(
            "[STATE] Updated fields: "
            + ", ".join(changed_fields)
        )

    else:
        print(
            "[STATE] No mission parameter changed."
        )

    if scheduled:
        print(
            "[ROUTER] Scheduled checks: "
            + " -> ".join(scheduled)
        )

    else:
        print(
            "[ROUTER] Reusing cached results."
        )

    return {
        **updates,
        "intent": "analyse",
        "updated_fields": changed_fields,
        "flight_result": flight_result,
        "wind_result": wind_result,
        "battery_result": battery_result,
        "run_flight": run_flight,
        "run_wind": run_wind,
        "run_battery": run_battery,
        "failure_reasons": [],
        "replan_actions": [],
        "replan_deferred_actions": [],
        "replan_requires_reevaluation": False,
        "replan_status": "",
        "response": "",
    }


def route_after_input(
    state: AutonomousMissionState,
) -> Literal[
    "planner",
    "trace",
    "question",
]:
    if state["intent"] == "trace":
        return "trace"

    if state["intent"] == "question":
        return "question"

    return "planner"


def mission_planner_node(
    state: AutonomousMissionState,
) -> dict:
    print(
        "[NODE] mission_planner"
    )

    current_version = state.get(
        "plan_version",
        0,
    )

    changed_fields = state.get(
        "updated_fields",
        [],
    )

    if (
        current_version == 0
        or changed_fields
    ):
        version = current_version + 1

    else:
        version = current_version

    plan = (
        f"Plan version: {version}\n"
        f"Objective: Fly "
        f"{state.get('distance_km', 'unknown')} km "
        f"towards "
        f"{bearing_name(state.get('flight_bearing_deg'))}.\n"
        f"Drone airspeed: "
        f"{state.get('drone_speed_kmh', 'unknown')} km/h.\n"
        f"Wind: "
        f"{state.get('wind_speed_kmh', 'unknown')} km/h "
        f"from "
        f"{bearing_name(state.get('wind_direction_from_deg'))}.\n"
        f"Starting battery: "
        f"{state.get('battery_percent', 'unknown')}%.\n"
        "Policy: validate flight dynamics, wind safety and "
        "battery safety; resolve upstream constraints before "
        "recalculating dependent requirements."
    )

    return {
        "plan_version": version,
        "mission_plan": plan,
    }


def route_after_planner(
    state: AutonomousMissionState,
) -> Literal[
    "flight",
    "wind",
    "battery",
    "evaluate",
]:
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

    return "evaluate"


def flight_check_node(
    state: AutonomousMissionState,
) -> dict:
    print(
        "[NODE] flight_check"
    )

    result = calculate_flight_time.invoke(
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

    return {
        "flight_result": result,
        "run_flight": False,
        "run_battery": (
            state.get(
                "run_battery",
                False,
            )
            and result.get("status")
            == "success"
        ),
    }


def route_after_flight(
    state: AutonomousMissionState,
) -> Literal[
    "wind",
    "battery",
    "evaluate",
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

    return "evaluate"


def wind_check_node(
    state: AutonomousMissionState,
) -> dict:
    print(
        "[NODE] wind_check"
    )

    result = check_wind_safety.invoke(
        {
            "wind_speed_kmh":
                state["wind_speed_kmh"],
            "max_safe_wind_speed_kmh":
                state[
                    "max_safe_wind_speed_kmh"
                ],
        }
    )

    return {
        "wind_result": result,
        "run_wind": False,
    }


def route_after_wind(
    state: AutonomousMissionState,
) -> Literal[
    "battery",
    "evaluate",
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

    return "evaluate"


def battery_check_node(
    state: AutonomousMissionState,
) -> dict:
    print(
        "[NODE] battery_check"
    )

    flight_result = state[
        "flight_result"
    ]

    result = check_battery.invoke(
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

    return {
        "battery_result": result,
        "run_battery": False,
    }


def evaluate_plan_node(
    state: AutonomousMissionState,
) -> dict:
    print(
        "[NODE] evaluate_plan"
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

    missing = get_missing_fields(
        state
    )

    invalid = False
    unsafe = False
    reasons = []

    for name, result in [
        ("Flight dynamics", flight_result),
        ("Wind safety", wind_result),
        ("Battery safety", battery_result),
    ]:
        if (
            result is not None
            and result.get("status")
            == "error"
        ):
            invalid = True

            reasons.append(
                f"{name}: "
                f"{result.get('message', 'validation error')}"
            )

    if (
        flight_result is not None
        and flight_result.get("status")
        == "unsafe"
    ):
        unsafe = True

        reasons.append(
            "Flight dynamics: "
            + flight_result.get(
                "message",
                "unsafe flight dynamics",
            )
        )

    if (
        wind_result is not None
        and wind_result.get("status")
        == "unsafe"
    ):
        unsafe = True

        reasons.append(
            "Wind safety: current wind exceeds "
            "the allowed safety limit."
        )

    if (
        battery_result is not None
        and battery_result.get("status")
        == "unsafe"
    ):
        unsafe = True

        reasons.append(
            "Battery safety: the drone would finish "
            "below the required reserve."
        )

    if invalid:
        status = "INVALID"

    elif unsafe:
        status = "NOT FEASIBLE"

    elif missing or (
        flight_result is None
        or wind_result is None
        or battery_result is None
    ):
        status = "UNDETERMINED"

    else:
        status = "FEASIBLE"

    print(
        "[EVALUATOR] Mission status: "
        f"{status}"
    )

    return {
        "mission_status": status,
        "missing_fields": missing,
        "failure_reasons": reasons,
    }


def route_after_evaluation(
    state: AutonomousMissionState,
) -> Literal[
    "replan",
    "report",
]:
    if (
        state.get(
            "mission_status"
        )
        == "NOT FEASIBLE"
    ):
        return "replan"

    return "report"


def replanner_node(
    state: AutonomousMissionState,
) -> dict:
    print(
        "[NODE] replanner"
    )

    actions = []
    deferred_actions = []

    requires_redesign = False
    requires_reevaluation = False

    flight_result = state.get(
        "flight_result"
    )

    wind_result = state.get(
        "wind_result"
    )

    battery_result = state.get(
        "battery_result"
    )

    flight_unsafe = (
        flight_result is not None
        and flight_result.get("status")
        == "unsafe"
    )

    wind_unsafe = (
        wind_result is not None
        and wind_result.get("status")
        == "unsafe"
    )

    battery_unsafe = (
        battery_result is not None
        and battery_result.get("status")
        == "unsafe"
    )

    if flight_unsafe:
        crosswind = flight_result.get(
            "crosswind_component_kmh"
        )

        relative_angle = math.radians(
            state["wind_direction_from_deg"]
            - state["flight_bearing_deg"]
        )

        crosswind_factor = abs(
            math.sin(relative_angle)
        )

        drone_speed = state.get(
            "drone_speed_kmh"
        )

        dynamic_wind_limit = None

        if (
            drone_speed is not None
            and drone_speed > 0
            and crosswind_factor > 1e-9
        ):
            dynamic_wind_limit = (
                drone_speed
                / crosswind_factor
            )

        if (
            wind_unsafe
            and dynamic_wind_limit is not None
        ):
            effective_limit = min(
                state[
                    "max_safe_wind_speed_kmh"
                ],
                dynamic_wind_limit,
            )

            actions.append(
                "Delay execution until wind speed is below "
                f"{effective_limit:.1f} km/h for the current "
                "route and wind direction, or redesign the route."
            )

        elif dynamic_wind_limit is not None:
            actions.append(
                "Reduce the crosswind conditions so wind speed "
                f"is below {dynamic_wind_limit:.1f} km/h for "
                "the current wind direction, or redesign the route."
            )

        elif crosswind is not None:
            actions.append(
                "Change the route or wait for different wind "
                "conditions until the drone can maintain the "
                "requested ground track."
            )

        else:
            actions.append(
                "Change the route or wait for wind conditions "
                "that produce safe flight dynamics."
            )

        deferred_actions.append(
            "Recalculate flight time and battery requirements "
            "after the route or wind conditions are updated."
        )

        requires_redesign = True
        requires_reevaluation = True

    elif wind_unsafe:
        actions.append(
            "Delay execution until wind speed is at or below "
            f"{state.get('max_safe_wind_speed_kmh')} km/h."
        )

        deferred_actions.append(
            "Recalculate flight dynamics and battery requirements "
            "after the wind speed is updated because both depend "
            "on the new environmental conditions."
        )

        requires_reevaluation = True

    elif battery_unsafe:
        required_start = (
            battery_result[
                "required_battery_percent"
            ]
            + state[
                "reserve_percent"
            ]
        )

        minimum_start = (
            math.ceil(
                required_start * 10
            )
            / 10
        )

        if minimum_start <= 100:
            actions.append(
                "Recharge or replace the battery so the mission "
                f"starts with at least {minimum_start}% battery."
            )

        else:
            actions.append(
                "The mission requires more than 100% battery "
                "under the current validated flight conditions. "
                "Split the mission into multiple stages, use a "
                "recharge/battery-swap stop, or redesign the route."
            )

            requires_redesign = True

    if not actions:
        actions.append(
            "No deterministic corrective action could be generated."
        )

    if requires_redesign:
        replan_status = (
            "MISSION REDESIGN OR ENVIRONMENTAL UPDATE REQUIRED"
        )

    elif requires_reevaluation:
        replan_status = (
            "ENVIRONMENTAL UPDATE AND RE-EVALUATION REQUIRED"
        )

    else:
        replan_status = "ACTION REQUIRED"

    print(
        "[REPLANNER] "
        f"{replan_status}"
    )

    if deferred_actions:
        print(
            "[REPLANNER] Dependent decisions deferred "
            "until upstream conditions are updated."
        )

    return {
        "replan_status": replan_status,
        "replan_actions": actions,
        "replan_deferred_actions": deferred_actions,
        "replan_requires_reevaluation": requires_reevaluation,
    }


def build_trace(
    state: AutonomousMissionState,
) -> str:
    steps = [
        (
            f"Create or update mission plan version "
            f"{state.get('plan_version', 1)}."
        ),
        "Read the current mission parameters.",
    ]

    flight = state.get(
        "flight_result"
    )

    wind = state.get(
        "wind_result"
    )

    battery = state.get(
        "battery_result"
    )

    if flight is not None:
        if "headwind_component_kmh" in flight:
            steps.append(
                "Headwind component: "
                f"{flight['headwind_component_kmh']} km/h."
            )

        if "crosswind_component_kmh" in flight:
            steps.append(
                "Crosswind component: "
                f"{flight['crosswind_component_kmh']} km/h."
            )

        if flight.get("status") == "success":
            steps.append(
                "Estimated flight time: "
                f"{flight['flight_time_minutes']} minutes."
            )

        else:
            steps.append(
                flight.get(
                    "message",
                    "Flight dynamics failed.",
                )
            )

    if wind is not None:
        steps.append(
            "Wind check result: "
            f"{wind.get('status')}."
        )

    if battery is not None:
        steps.append(
            "Battery check result: "
            f"{battery.get('status')}."
        )

    steps.append(
        "Mission evaluation result: "
        f"{state.get('mission_status', 'UNDETERMINED')}."
    )

    for action in state.get(
        "replan_actions",
        [],
    ):
        steps.append(
            "Immediate replan action: "
            f"{action}"
        )

    for action in state.get(
        "replan_deferred_actions",
        [],
    ):
        steps.append(
            "Deferred dependent decision: "
            f"{action}"
        )

    if state.get(
        "replan_requires_reevaluation",
        False,
    ):
        steps.append(
            "The mission must be re-evaluated after the upstream "
            "condition changes before dependent recommendations "
            "are considered final."
        )

    return "\n".join(
        f"{number}. {step}"
        for number, step
        in enumerate(
            steps,
            start=1,
        )
    )


def final_report_node(
    state: AutonomousMissionState,
) -> dict:
    print(
        "[NODE] final_report"
    )

    output = [
        "AUTONOMOUS MISSION PLAN",
        state.get(
            "mission_plan",
            "No mission plan is available.",
        ),
        "\nMISSION EVALUATION",
    ]

    flight = state.get(
        "flight_result"
    )

    wind = state.get(
        "wind_result"
    )

    battery = state.get(
        "battery_result"
    )

    if flight is None:
        output.append(
            "Flight dynamics: UNDETERMINED"
        )

    elif flight["status"] == "success":
        output.extend(
            [
                "Flight dynamics: SAFE",
                (
                    "Estimated flight time: "
                    f"{flight['flight_time_minutes']} minutes"
                ),
                (
                    "Estimated ground speed: "
                    f"{flight['ground_speed_kmh']} km/h"
                ),
                (
                    "Headwind component: "
                    f"{flight['headwind_component_kmh']} km/h"
                ),
                (
                    "Crosswind component: "
                    f"{flight['crosswind_component_kmh']} km/h"
                ),
            ]
        )

    else:
        output.extend(
            [
                (
                    "Flight dynamics: "
                    f"{flight['status'].upper()}"
                ),
                flight.get(
                    "message",
                    "Unknown flight result.",
                ),
            ]
        )

    if wind is None:
        output.append(
            "Wind safety: UNDETERMINED"
        )

    elif wind["status"] == "safe":
        output.append(
            "Wind safety: SAFE "
            f"({wind['wind_speed_kmh']} km/h <= "
            f"{wind['max_safe_wind_speed_kmh']} km/h)"
        )

    elif wind["status"] == "unsafe":
        output.append(
            "Wind safety: UNSAFE "
            f"({wind['wind_speed_kmh']} km/h > "
            f"{wind['max_safe_wind_speed_kmh']} km/h)"
        )

    else:
        output.extend(
            [
                "Wind safety: ERROR",
                wind.get(
                    "message",
                    "Unknown wind result.",
                ),
            ]
        )

    if battery is None:
        output.append(
            "Battery safety: UNDETERMINED"
        )

    elif battery["status"] in {
        "safe",
        "unsafe",
    }:
        output.extend(
            [
                (
                    "Battery safety: "
                    f"{battery['status'].upper()}"
                ),
                (
                    "Estimated battery consumption: "
                    f"{battery['required_battery_percent']}%"
                ),
                (
                    "Estimated battery after flight: "
                    f"{battery['battery_after_flight_percent']}%"
                ),
                (
                    "Required battery reserve: "
                    f"{battery['required_reserve_percent']}%"
                ),
            ]
        )

    else:
        output.extend(
            [
                "Battery safety: ERROR",
                battery.get(
                    "message",
                    "Unknown battery result.",
                ),
            ]
        )

    status = state.get(
        "mission_status",
        "UNDETERMINED",
    )

    output.append(
        "\nFinal mission feasibility: "
        f"{status}"
    )

    missing = state.get(
        "missing_fields",
        [],
    )

    if missing:
        output.append(
            "\nMissing information:"
        )

        for item in missing:
            output.append(
                f"- {item}"
            )

    reasons = state.get(
        "failure_reasons",
        [],
    )

    if reasons:
        output.append(
            "\nFailure reasons:"
        )

        for reason in reasons:
            output.append(
                f"- {reason}"
            )

    actions = state.get(
        "replan_actions",
        [],
    )

    deferred_actions = state.get(
        "replan_deferred_actions",
        [],
    )

    if actions:
        output.append(
            "\nAUTONOMOUS REPLAN PROPOSAL"
        )

        output.append(
            "Replan status: "
            f"{state.get('replan_status')}"
        )

        output.append(
            "\nImmediate action:"
        )

        for index, action in enumerate(
            actions,
            start=1,
        ):
            output.append(
                f"{index}. {action}"
            )

        if deferred_actions:
            output.append(
                "\nDeferred dependent decisions:"
            )

            for index, action in enumerate(
                deferred_actions,
                start=1,
            ):
                output.append(
                    f"{index}. {action}"
                )

        if state.get(
            "replan_requires_reevaluation",
            False,
        ):
            output.append(
                "\nThe planner intentionally does not finalize "
                "downstream requirements yet. Update the upstream "
                "condition first, then run the mission again so "
                "dependent values are recalculated from the new state."
            )

        output.append(
            "\nThe proposed actions have not been physically "
            "applied. Update the mission state after the action "
            "has actually occurred."
        )

    elif status == "FEASIBLE":
        output.append(
            "\nPlanner decision: mission can proceed "
            "under the current assumptions."
        )

    elif status == "INVALID":
        output.append(
            "\nPlanner decision: correct the invalid data "
            "before replanning."
        )

    elif status == "UNDETERMINED":
        output.append(
            "\nPlanner decision: provide the missing information "
            "before execution."
        )

    report = "\n".join(
        output
    )

    temporary_state = {
        **state,
        "final_report": report,
    }

    trace = build_trace(
        temporary_state
    )

    return {
        "final_report": report,
        "calculation_trace": trace,
        "response": report,
    }


def show_trace_node(
    state: AutonomousMissionState,
) -> dict:
    print(
        "[NODE] show_trace"
    )

    trace = state.get(
        "calculation_trace"
    )

    if not trace:
        response = (
            "No mission has been planned or analysed yet."
        )

    else:
        response = (
            "Planning and calculation steps:\n"
            + trace
        )

    return {
        "response": response,
    }


def answer_question_node(
    state: AutonomousMissionState,
) -> dict:
    print(
        "[NODE] answer_question"
    )

    print(
        "[AGENT] Thinking..."
    )

    system_prompt = f"""
You are an autonomous drone mission planning assistant.

Answer using only the mission state, current mission plan,
deterministic analysis, replan proposal, and calculation trace below.

Never invent or modify mission parameters.
Never invent calculation results.
Never claim a physical action has happened unless the state says so.
Do not perform new safety calculations yourself.
If information is unavailable, say so.

Do not reveal hidden reasoning.
Only provide the final concise explanation.

CURRENT MISSION STATE:
{state_to_text(state)}

CURRENT PLAN:
{state.get('mission_plan', 'No plan available.')}

LAST REPORT:
{state.get('final_report', 'No report available.')}

CALCULATION TRACE:
{state.get('calculation_trace', 'No trace available.')}

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
    AutonomousMissionState
)

builder.add_node(
    "process_input",
    process_input_node,
)

builder.add_node(
    "mission_planner",
    mission_planner_node,
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
    "evaluate_plan",
    evaluate_plan_node,
)

builder.add_node(
    "replanner",
    replanner_node,
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
        "planner":
            "mission_planner",
        "trace":
            "show_trace",
        "question":
            "answer_question",
    },
)

builder.add_conditional_edges(
    "mission_planner",
    route_after_planner,
    {
        "flight":
            "flight_check",
        "wind":
            "wind_check",
        "battery":
            "battery_check",
        "evaluate":
            "evaluate_plan",
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
        "evaluate":
            "evaluate_plan",
    },
)

builder.add_conditional_edges(
    "wind_check",
    route_after_wind,
    {
        "battery":
            "battery_check",
        "evaluate":
            "evaluate_plan",
    },
)

builder.add_edge(
    "battery_check",
    "evaluate_plan",
)

builder.add_conditional_edges(
    "evaluate_plan",
    route_after_evaluation,
    {
        "replan":
            "replanner",
        "report":
            "final_report",
    },
)

builder.add_edge(
    "replanner",
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

autonomous_mission_graph = (
    builder.compile(
        checkpointer=memory
    )
)


THREAD_ID = (
    "autonomous-drone-mission-session"
)

config = {
    "configurable": {
        "thread_id":
            THREAD_ID
    }
}


def main() -> None:
    print(
        "LangGraph Autonomous Drone Mission Planner"
    )

    print(
        "\nDescribe the drone mission."
        "\nThe planner will analyse, evaluate and replan when necessary."
        "\nYou can update mission parameters in later messages."
        "\nYou can also ask questions about the current plan."
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
                "\n\nClosing Autonomous Drone Mission Planner..."
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
                "\nClosing Autonomous Drone Mission Planner..."
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
            "\n[LANGGRAPH] Running autonomous planning graph..."
        )

        try:
            result = (
                autonomous_mission_graph.invoke(
                    {
                        "user_input":
                            user_input
                    },
                    config=config,
                )
            )

        except Exception as error:  # noqa: BLE001
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
