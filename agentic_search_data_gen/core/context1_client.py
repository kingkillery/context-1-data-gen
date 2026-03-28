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
        return response.json()

    def _iter_sse(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            yield json.loads(data)
