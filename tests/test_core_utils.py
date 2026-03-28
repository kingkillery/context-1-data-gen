import os
import unittest
from unittest.mock import patch

from anthropic import Anthropic

from agentic_search_data_gen.core.context1_client import Context1Client
from agentic_search_data_gen.core.utils import (
    DEFAULT_CONTEXT1_BASE_URL,
    DEFAULT_FRONTIER_WS_URL,
    get_anthropic_client,
    get_context1_base_url,
    get_frontier_ws_url,
)


class ProviderConfigTests(unittest.TestCase):
    def test_minimax_client_uses_anthropic_compatible_base_url(self):
        with patch.dict(
            os.environ,
            {
                "MINIMAX_API_KEY": "minimax-test",
                "MINIMAX_BASE_URL": "https://example.minimax.local/v1",
            },
            clear=False,
        ):
            client = get_anthropic_client()
            self.assertIsInstance(client, Anthropic)
            self.assertEqual(client.api_key, "minimax-test")
            self.assertEqual(str(client._client.base_url), "https://example.minimax.local/v1/")
            client.close()

    def test_anthropic_fallback_works_without_minimax(self):
        env = dict(os.environ)
        env.pop("MINIMAX_API_KEY", None)
        env.pop("MINIMAX_BASE_URL", None)
        env["ANTHROPIC_API_KEY"] = "anthropic-test"
        with patch.dict(
            os.environ,
            env,
            clear=True,
        ):
            client = get_anthropic_client()
            self.assertIsInstance(client, Anthropic)
            self.assertEqual(client.api_key, "anthropic-test")
            client.close()

    def test_context1_and_frontier_defaults_are_stable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_context1_base_url(), DEFAULT_CONTEXT1_BASE_URL)
            self.assertEqual(get_frontier_ws_url(), DEFAULT_FRONTIER_WS_URL)

    def test_context1_client_defaults_to_public_service_url(self):
        with patch.dict(os.environ, {}, clear=True):
            client = Context1Client()
        self.assertEqual(client.base_url, DEFAULT_CONTEXT1_BASE_URL)
