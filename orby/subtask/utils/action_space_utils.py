from typing import Literal, TypedDict

from orby.subtask.utils.action_parsing_utils import extract_action

VALID_ACTION_TYPES = [
    "click",
    "complete",
    "drag_and_release",
    "hover",
    "key_press",
    "scroll",
    "type",
    "wait",
]


class ActionInfo(TypedDict):
    action_type: Literal[
        "click",
        "complete",
        "drag_and_release",
        "hover",
        "key_press",
        "scroll",
        "type",
        "wait",
    ]
    coordinates: list[tuple[float, float]] | None
    args: dict[str, str] | None


def get_action_info(action: str) -> ActionInfo:
    action_type = extract_action(action)
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f"Invalid action type: {action_type}")
    return eval(action)


def click(x: float, y: float, button: Literal["left", "right"] = "left", double: bool = False) -> ActionInfo:
    return ActionInfo(
        action_type="click",
        coordinates=[(x, y)],
        args={"button": str(button), "double": str(double)},
    )


def complete(answer: str = "", infeasible_reason: str = "") -> ActionInfo:
    return ActionInfo(
        action_type="complete",
        coordinates=None,
        args={"answer": str(answer), "infeasible_reason": str(infeasible_reason)},
    )


def drag_and_release(x1: float, y1: float, x2: float, y2: float) -> ActionInfo:
    return ActionInfo(
        action_type="drag_and_release",
        coordinates=[(x1, y1), (x2, y2)],
        args=None,
    )


def hover(x: float, y: float) -> ActionInfo:
    return ActionInfo(
        action_type="hover",
        coordinates=[(x, y)],
        args=None,
    )


def key_press(keys: list[str]) -> ActionInfo:
    return ActionInfo(
        action_type="key_press",
        coordinates=None,
        args={"keys": str(keys)},
    )


def scroll(x: float, y: float, delta_x: float = 0, delta_y: float = 100) -> ActionInfo:
    return ActionInfo(
        action_type="scroll",
        coordinates=[(x, y)],
        args={
            "horizontal": str("left" if delta_x < 0 else "right"),
            "vertical": str("up" if delta_y < 0 else "down"),
        },
    )


def type(x: float, y: float, text: str) -> ActionInfo:
    return ActionInfo(
        action_type="type",
        coordinates=[(x, y)],
        args={"text": str(text)},
    )


def wait(ms: int = 1000) -> ActionInfo:
    return ActionInfo(
        action_type="wait",
        coordinates=None,
        args={"ms": str(ms)},
    )
