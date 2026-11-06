from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from typing import Any


def mapping_candidates(text: str) -> Iterable[str]:
    start: int | None = None
    depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if depth and character in ('"', "'"):
            quote = character
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
                start = None


def extract_list_field(text: str, field: str) -> list[Any]:
    for candidate in mapping_candidates(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(value, dict) and isinstance(value.get(field), list):
            return value[field]
    raise ValueError(f"vLLM response has no parseable {field} list")
