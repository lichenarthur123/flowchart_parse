"""OpenAI-compatible model transport."""
from __future__ import annotations
import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional
import httpx
from pydantic import BaseModel, Field
from usage_tracking import record_request, response_usage


class ModelConfig(BaseModel):
    base_url: str
    model: str
    api_key: Optional[str] = Field(default=None, repr=False, exclude=True)
    api_key_env: Optional[str] = None
    timeout_seconds: float = Field(default=120, gt=0)
    max_tokens: int = Field(default=4096, gt=0)
    json_mode: bool = False
    enable_thinking: Optional[bool] = None


class ModelClient:
    """One transport for all providers. Three attempts only for transient failures."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.key = (os.environ.get(config.api_key_env) if config.api_key_env else None) or config.api_key
        if not self.key:
            raise ValueError(f"模型 {config.model} 未配置 API Key（api_key 或 api_key_env）。")

    def _request(self, prompt: str, images: list[Path], json_output: bool, stage: str) -> str:
        content = [{"type": "text", "text": prompt + ("\n只返回一个 JSON 对象，不要 Markdown。" if json_output else "")}]
        for path in images:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        cfg = self.config
        url = cfg.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload = {"model": cfg.model, "messages": [{"role": "user", "content": content}],
                   "max_tokens": cfg.max_tokens}
        if json_output and cfg.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if cfg.enable_thinking is not None:
            payload["enable_thinking"] = cfg.enable_thinking
        call_id = uuid.uuid4().hex
        with httpx.Client(timeout=cfg.timeout_seconds) as client:
            for attempt in range(3):
                started = time.monotonic()
                try:
                    response = client.post(url, headers={"Authorization": f"Bearer {self.key}"}, json=payload)
                except httpx.TransportError as exc:
                    record_request({'call_id': call_id, 'model': cfg.model, 'stage': stage,
                                    'attempt': attempt+1, 'http_status': None,
                                    'duration_ms': round((time.monotonic()-started)*1000),
                                    'error_type': type(exc).__name__, 'usage': {}})
                    if attempt == 2:
                        raise RuntimeError("模型连接失败或超时，已尝试 3 次。") from None
                    time.sleep(2 ** attempt)
                    continue
                record_request({'call_id': call_id, 'model': cfg.model, 'stage': stage,
                                'attempt': attempt+1, 'http_status': response.status_code,
                                'duration_ms': round((time.monotonic()-started)*1000),
                                'usage': response_usage(response)})
                if response.status_code in (408, 429) or response.status_code >= 500:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                if response.is_error:
                    raise RuntimeError(f"模型请求失败：HTTP {response.status_code}；检查地址、权限及配额。")
                break
        try:
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") == "length":
                raise RuntimeError("模型输出被截断：请缩小处理范围或提高 max_tokens。")
            raw = choice["message"]["content"].strip()
            return raw
        except (ValueError, KeyError, IndexError, AttributeError, TypeError):
            raise RuntimeError("模型返回格式无效：需要 Chat Completions 响应。") from None

    def ask_text(self, prompt: str, images: list[Path], *, stage: str = 'unspecified') -> str:
        """Return raw text for OCR endpoints that do not produce JSON."""
        return self._request(prompt, images, json_output=False, stage=stage)

    def ask_json(self, prompt: str, images: list[Path], *, stage: str = 'unspecified') -> dict:
        for attempt in range(3):
            raw = self._request(prompt, images, json_output=True, stage=stage)
            try:
                if raw.startswith("```") and raw.endswith("```"):
                    raw = "\n".join(raw.splitlines()[1:-1])
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object")
                return data
            except (ValueError, KeyError, IndexError, AttributeError, TypeError):
                if attempt == 2:
                    raise RuntimeError("模型返回格式无效：需要 Chat Completions 响应及 JSON 对象内容，已尝试 3 次。") from None
                time.sleep(2 ** attempt)


