"""Record actual model requests and token usage without prompts or credentials."""
from __future__ import annotations

import json
import os
from pathlib import Path

ENV_VAR = 'FLOWCHART_USAGE_LOG'


def record_request(event: dict) -> None:
    path = os.environ.get(ENV_VAR)
    if path:
        with Path(path).open('a', encoding='utf-8') as file:
            file.write(json.dumps(event, ensure_ascii=False, separators=(',', ':')) + '\n')


def response_usage(response) -> dict:
    try:
        data = response.json()
        usage = data.get('usage', {}) if isinstance(data, dict) else {}
        if not isinstance(usage, dict):
            return {}
    except (ValueError, AttributeError):
        return {}
    result = {}
    for name in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'input_tokens', 'output_tokens'):
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[name] = value
    for name in ('prompt_tokens_details', 'completion_tokens_details', 'input_tokens_details', 'output_tokens_details'):
        details = usage.get(name)
        if isinstance(details, dict):
            result[name] = {k: v for k, v in details.items()
                            if isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 0}
    return result
