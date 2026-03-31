"""Helpers for talking to the hosted Context-1 service from a local harness."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional

import requests

from .utils import get_context1_base_url


class Context1Client:
    """Thin REST client for the hosted Context-1 agent service."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 60):
        self.base_url = (base_url or get_context1_base_url()).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def messages(self):
        """Mock the Anthropic messages interface."""
        return self

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 1024,
        temperature: float = 0,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """Anthropic-compatible create method that maps to the Context-1 API."""
        trajectory = []
        if system:
            trajectory.append({"role": "system", "content": system})
        trajectory.extend(messages)

        payload = {
            "trajectory": trajectory,
            "tools": tools or [],
            "generation": {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "stream": stream,
        }
        return self.agent_step(payload, stream=stream)

    def healthz(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/healthz", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def agent_step(
        self,
        payload: Dict[str, Any],
        *,
        stream: bool = False,
        timeout: Optional[int] = None,
    ) -> Any:
        request_timeout = timeout or self.timeout
        response = self.session.post(
            f"{self.base_url}/v1/agent/step",
            json=payload,
            timeout=request_timeout,
            stream=stream,
        )
        response.raise_for_status()
        if stream:
            return self._iter_sse(response)
        
        # Wrap JSON response in a mock object to match Anthropic Message response
        data = response.json()
        return self._wrap_response(data)

    def _wrap_response(self, data: Dict[str, Any]) -> Any:
        """Wrap raw API response into an Anthropic-like Message object."""
        from types import SimpleNamespace
        
        # If it's already an Anthropic-like structure, just return it
        if "content" in data:
            return SimpleNamespace(
                content=[SimpleNamespace(**c) if isinstance(c, dict) else c for c in data["content"]]
            )
        
        # Otherwise, assume it's a standard Chat Completion response and map it
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        wrapped_content = []
        if content:
            wrapped_content.append(SimpleNamespace(type="text", text=content))
        
        for tc in tool_calls:
            # Map OpenAI/vLLM tool call to Anthropic tool_use
            wrapped_content.append(SimpleNamespace(
                type="tool_use",
                id=tc.get("id"),
                name=tc.get("function", {}).get("name"),
                input=json.loads(tc.get("function", {}).get("arguments", "{}"))
            ))
            
        return SimpleNamespace(content=wrapped_content)

    def _iter_sse(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue
