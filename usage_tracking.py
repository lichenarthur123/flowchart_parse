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
            result[name] = {k: v for k,v in details.items() if isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 0}
    return result


def summarize(log: Path) -> dict:
    events = []
    if log.is_file():
        for number, line in enumerate(log.read_text(encoding='utf-8').splitlines(), 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError()
                events.append(event)
            except ValueError:
                raise ValueError(f'模型用量日志第 {number} 行无效。') from None

    def group(items):
        ids = {e['call_id'] for e in items}
        values = {'input': [], 'output': [], 'total': []}
        missing = {'input': 0, 'output': 0, 'total': 0}
        image_tokens, input_text_tokens = [], []
        for e in items:
            usage = e.get('usage') or {}
            details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
            for key, target in (('image_tokens', image_tokens), ('text_tokens', input_text_tokens)):
                value = details.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    target.append(value)
            counts = {'input': usage.get('prompt_tokens', usage.get('input_tokens')),
                      'output': usage.get('completion_tokens', usage.get('output_tokens')),
                      'total': usage.get('total_tokens')}
            for key, value in counts.items():
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    values[key].append(value)
                else:
                    missing[key] += 1
        return {'model_calls': len(ids), 'http_requests': len(items),
                'successful_http_responses': sum(isinstance(e.get('http_status'), int) and 200 <= e['http_status'] < 300 for e in items),
                'failed_or_unknown_requests': sum(not isinstance(e.get('http_status'), int) or not 200 <= e['http_status'] < 300 for e in items),
                'reported_input_tokens': sum(values['input']) if values['input'] else None,
                'reported_output_tokens': sum(values['output']) if values['output'] else None,
                'reported_total_tokens': sum(values['total']) if values['total'] else None,
                'reported_image_tokens': sum(image_tokens) if image_tokens else None,
                'reported_input_text_tokens': sum(input_text_tokens) if input_text_tokens else None,
                'requests_missing_image_token_count': len(items)-len(image_tokens),
                'requests_missing_input_text_token_count': len(items)-len(input_text_tokens),
                'requests_missing_token_counts': missing}

    result = {'total': group(events), 'by_model': {}, 'by_stage': {},
              'note': '仅汇总接口实际返回的 usage；缺失数据不按零计算。模型调用按逻辑调用计数，重试另计 HTTP 请求。'}
    for key, field in (('by_model', 'model'), ('by_stage', 'stage')):
        names = sorted({e.get(field, 'unknown') for e in events})
        result[key] = {name: group([e for e in events if e.get(field, 'unknown') == name]) for name in names}
    return result
