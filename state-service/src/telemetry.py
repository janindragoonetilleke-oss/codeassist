from ipaddress import ip_address
import os
import json
import requests
from datetime import datetime, timezone
import statistics
import dataclasses
from typing import List, Dict, Any, Optional

from src.config import settings
from src.api.datatypes import EpisodeSession, ActionIndex
from src.store.schema import Episode
from src.utils import _load_problem
import logging

logger = logging.getLogger(__name__)

import platform


TELEMETRY_API_EPISODE_SESSION = (
    f"{settings.TELEMETRY_BASE_URL}/event/codeassist/episode"
)

CODEASSIST_VERSION = os.environ.get("CODEASSIST_VERSION", "unknown")


def _load_problem_question_id(problem_id: str) -> int | None:
    """Load the question_id (numeric ID) for a given problem_id (task_id string).

    Args:
        problem_id: The task_id string (e.g., "two-sum")

    Returns:
        The numeric question_id if found, None otherwise
    """
    problem = _load_problem(problem_id)
    if problem:
        question_id = problem.get("question_id")
        if isinstance(question_id, int):
            return question_id
    return None


def get_system_info():
    return {
        "uname": json.dumps(platform.uname()._asdict()),
        "arch": platform.machine(),
        "os": platform.system(),
        "accelerators": get_accelerator_info(),
        "ip": get_ip(),
    }


def get_ip():
    ip = requests.get("https://icanhazip.com/", timeout=1).text
    return ip


def get_accelerator_info():
    import torch

    out_devices = []

    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(device)
            d = {
                "name": properties.name,
                "major": properties.major,
                "minor": properties.minor,
                "total_memory": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
                "max_threads_per_multi_processor": properties.max_threads_per_multi_processor,
            }
            out_devices.append(d)

    return out_devices


def is_telemetry_disabled():
    return os.environ.get("DISABLE_TELEMETRY", "false").lower() in ("true", "1", "yes")


def get_user_id() -> str:
    with open(
        os.path.join(settings.PERSISTENT_DATA_DIR, "auth/userKeyMap.json"), "r"
    ) as f:
        user_key_map = json.load(f)
        keys = list(user_key_map.keys())

        for key in keys:
            return user_key_map[key]["user"]["accountAddress"]

        return "unknown"


def _is_assistant_action(action_dict: Optional[dict]) -> bool:
    if action_dict is None:
        return False
    else:
        if action_dict.get("A") is not None:
            return True
        else:
            return False


def push_telemetry_event_session(episode: Episode):
    if is_telemetry_disabled():
        return

    episode_session = convert_episode_session_to_telemetry_event(episode)

    try:
        ret = requests.post(
            TELEMETRY_API_EPISODE_SESSION, json=episode_session.model_dump(), timeout=2
        )
        ret.raise_for_status()
        logger.info(f"Pushed telemetry event session: {episode_session}")
    except Exception as e:
        logger.error(
            f"Error pushing telemetry event session: {e}, episode_session: {episode_session}"
        )


def _compute_action_statistics(episode: Episode) -> Dict[str, Any]:
    """Compute action-related statistics from episode states."""
    assistant_actions = {action: 0 for action in ActionIndex}
    human_actions = {action: 0 for action in ActionIndex}

    # Track distances for edit operations
    edit_existing_distances = []
    explain_single_distances = []
    explain_multi_distances = []

    for state in episode.states:
        if state.action is None:
            continue

        # Determine if this is assistant or human action based on attribution
        is_assistant = _is_assistant_action(state.action)

        # TODO: We will need to infer human actions in state service for this to work correctly
        action_type = (
            state.action.get("A").get("type")
            if is_assistant
            else state.action.get("H").get("type")
        )

        try:
            action_enum = ActionIndex(action_type)
            if is_assistant:
                assistant_actions[action_enum] += 1
            else:
                human_actions[action_enum] += 1

            # Calculate distances for specific action types
            if action_enum in [
                ActionIndex.EDIT_EXISTING_LINES,
                ActionIndex.EXPLAIN_SINGLE_LINES,
                ActionIndex.EXPLAIN_MULTI_LINE,
            ]:
                distance = _calculate_cursor_distance(state)
                if action_enum == ActionIndex.EDIT_EXISTING_LINES:
                    edit_existing_distances.append(distance)
                elif action_enum == ActionIndex.EXPLAIN_SINGLE_LINES:
                    explain_single_distances.append(distance)
                elif action_enum == ActionIndex.EXPLAIN_MULTI_LINE:
                    explain_multi_distances.append(distance)

        except (ValueError, TypeError):
            # Invalid action type, skip
            continue

    return {
        "assistant_actions": assistant_actions,
        "human_actions": human_actions,
        "edit_existing_distances": edit_existing_distances,
        "explain_single_distances": explain_single_distances,
        "explain_multi_distances": explain_multi_distances,
    }


def _calculate_cursor_distance(state) -> float:
    """Calculate distance from cursor to the target line of an action."""
    if not state.action:
        return 0.0

    target_line = state.action.get("target_line", 1)
    cursor_line = _get_cursor_line_from_attribution(state.attribution)

    if cursor_line is None:
        return 0.0

    return abs(target_line - cursor_line)


def _get_cursor_line_from_attribution(attribution: List[Dict]) -> int:
    """Extract cursor line from attribution data."""
    if not attribution:
        return None

    for attr in attribution:
        if isinstance(attr, dict) and "cursor" in attr:
            cursor_info = attr["cursor"]
            if isinstance(cursor_info, dict) and "line" in cursor_info:
                return cursor_info["line"]
            elif isinstance(cursor_info, dict) and "char" in cursor_info:
                # If we have character position, estimate line (rough approximation)
                char_pos = cursor_info.get("char", 0)
                return max(1, char_pos // 80)  # Assume ~80 chars per line
    return None


def _compute_percentile(data: List[float], percentile: float) -> float:
    """Compute percentile of a list of numbers."""
    if not data:
        return 0.0
    try:
        return statistics.quantiles(data, n=100)[int(percentile) - 1]
    except (IndexError, statistics.StatisticsError):
        return 0.0


def _compute_regression_rates(episode: Episode) -> Dict[str, float]:
    """Compute test and compile regression rates."""
    if len(episode.states) < 2:
        return {
            "test_regression_rate": 0.0,
            "compile_regression_rate": 0.0,
            "test_progression_rate": 0.0,
            "compile_progression_rate": 0.0,
        }

    test_regressions = 0
    compile_regressions = 0
    test_progressions = 0
    compile_progressions = 0
    total_transitions = 0

    for i in range(1, len(episode.states)):
        prev_state = episode.states[i - 1]
        curr_state = episode.states[i]

        prev_compiled = prev_state.env.get("compiled", False)
        curr_compiled = curr_state.env.get("compiled", False)

        prev_tests_passed = prev_state.env.get("tests", {}).get("passed", 0)
        curr_tests_passed = curr_state.env.get("tests", {}).get("passed", 0)

        # Check compile regression/progression
        if prev_compiled and not curr_compiled:
            compile_regressions += 1
        elif not prev_compiled and curr_compiled:
            compile_progressions += 1

        # Check test regression/progression
        if curr_tests_passed < prev_tests_passed:
            test_regressions += 1
        elif curr_tests_passed > prev_tests_passed:
            test_progressions += 1

        total_transitions += 1

    return {
        "test_regression_rate": test_regressions / max(total_transitions, 1),
        "compile_regression_rate": compile_regressions / max(total_transitions, 1),
        "test_progression_rate": test_progressions / max(total_transitions, 1),
        "compile_progression_rate": compile_progressions / max(total_transitions, 1),
    }


def _compute_latency_stats(episode: Episode) -> Dict[str, int]:
    """Compute latency percentiles from episode states."""
    if len(episode.states) < 2:
        return {
            "p50_latency_ms": 0,
            "p90_latency_ms": 0,
            "p99_latency_ms": 0,
        }

    latencies = []
    states = sorted(episode.states, key=lambda s: s.timestep)
    for i in range(1, len(states)):
        prev_ts = getattr(states[i - 1], "timestamp_ms", None)
        curr_ts = getattr(states[i], "timestamp_ms", None)
        if isinstance(prev_ts, int) and isinstance(curr_ts, int):
            delta = curr_ts - prev_ts
            if delta > 0:  # Only include positive latencies
                latencies.append(delta)

    if not latencies:
        return {
            "p50_latency_ms": 0,
            "p90_latency_ms": 0,
            "p99_latency_ms": 0,
        }

    latencies.sort()
    n = len(latencies)

    return {
        "p50_latency_ms": latencies[int(n * 0.5)] if n > 0 else 0,
        "p90_latency_ms": latencies[int(n * 0.9)] if n > 0 else 0,
        "p99_latency_ms": latencies[int(n * 0.99)] if n > 0 else 0,
    }


def convert_episode_session_to_telemetry_event(episode: Episode) -> EpisodeSession:
    timestamp = datetime.now(timezone.utc).isoformat()
    duration_ms = episode.end_time - episode.start_time
    total_turns = len(episode.states)
    user_id = get_user_id()
    # Look up the numeric question_id from the dataset based on the problem_id (task_id)
    question_id = _load_problem_question_id(episode.problem_id)
    try:
        ip_addr = get_ip()
    except Exception as e:
        ip_addr = None

    # Determine success based on final state
    final_state = (
        max(episode.states, key=lambda s: s.timestep) if episode.states else None
    )
    success = False
    time_to_pass = None
    turns_to_pass = None

    if final_state:
        # Success if it compiles and passes tests
        compiled = final_state.env.get("compiled", False)
        tests_passed = final_state.env.get("tests", {}).get("passed", 0)
        success = compiled and tests_passed > 0

        # Find when it first passed (if it did)
        # TODO: pass probably means passed all tests, this is temporary
        for state in sorted(episode.states, key=lambda s: s.timestep):
            if (
                state.env.get("compiled", False)
                and state.env.get("tests", {}).get("passed", 0) > 0
            ):
                time_to_pass = max(
                    0, int(getattr(state, "timestamp_ms", 0)) - int(episode.start_time)
                )
                turns_to_pass = state.timestep
                break

    # Compute all statistics
    action_stats = _compute_action_statistics(episode)
    regression_rates = _compute_regression_rates(episode)
    latency_stats = _compute_latency_stats(episode)

    # Extract action counts
    assistant_actions = action_stats["assistant_actions"]
    human_actions = action_stats["human_actions"]

    return EpisodeSession(
        timestamp=timestamp,
        duration_ms=duration_ms,
        total_turns=total_turns,
        user_id=user_id,
        question_id=question_id,
        ip_addr=ip_addr,
        codeassist_version=CODEASSIST_VERSION,
        success=success,
        time_to_pass=time_to_pass,
        turns_to_pass=turns_to_pass,
        test_regression_rate=regression_rates["test_regression_rate"],
        compile_regression_rate=regression_rates["compile_regression_rate"],
        test_progression_rate=regression_rates["test_progression_rate"],
        compile_progression_rate=regression_rates["compile_progression_rate"],
        p50_latency_ms=latency_stats["p50_latency_ms"],
        p90_latency_ms=latency_stats["p90_latency_ms"],
        p99_latency_ms=latency_stats["p99_latency_ms"],
        assistant_noop_count=assistant_actions[ActionIndex.NO_OP],
        assistant_fill_partial_count=assistant_actions[ActionIndex.FILL_PARTIAL_LINE],
        assistant_write_single_count=assistant_actions[
            ActionIndex.REPLACE_AND_APPEND_SINGLE_LINE
        ],
        assistant_write_multi_count=assistant_actions[
            ActionIndex.REPLACE_AND_APPEND_MULTI_LINE
        ],
        assistant_edit_existing_count=assistant_actions[
            ActionIndex.EDIT_EXISTING_LINES
        ],
        assistant_explain_single_count=assistant_actions[
            ActionIndex.EXPLAIN_SINGLE_LINES
        ],
        assistant_explain_multi_count=assistant_actions[ActionIndex.EXPLAIN_MULTI_LINE],
        human_noop_count=human_actions[ActionIndex.NO_OP],
        human_fill_partial_count=human_actions[ActionIndex.FILL_PARTIAL_LINE],
        human_write_single_count=human_actions[
            ActionIndex.REPLACE_AND_APPEND_SINGLE_LINE
        ],
        human_write_multi_count=human_actions[
            ActionIndex.REPLACE_AND_APPEND_MULTI_LINE
        ],
        human_edit_existing_count=human_actions[ActionIndex.EDIT_EXISTING_LINES],
        human_explain_single_count=human_actions[ActionIndex.EXPLAIN_SINGLE_LINES],
        human_explain_multi_count=human_actions[ActionIndex.EXPLAIN_MULTI_LINE],
        episode_id=episode.episode_id,
    )
