import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

from agentic_search_data_gen.core.context1_client import Context1Client


class _Context1TestHandler(BaseHTTPRequestHandler):
    last_request = None

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path != "/healthz":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "status": "ok",
                    "model": "context-1",
                    "provider": "unit-test",
                }
            ).encode("utf-8")
        )

    def do_POST(self):
        if self.path != "/v1/agent/step":
            self.send_error(404)
            return

        content_length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        _Context1TestHandler.last_request = {
            "path": self.path,
            "headers": dict(self.headers.items()),
            "body": json.loads(body),
        }

        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "ok": True,
                    "received": _Context1TestHandler.last_request["body"],
                }
            ).encode("utf-8")
        )


class Context1ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Context1TestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_healthz_returns_json_from_configured_base_url(self):
        client = Context1Client(base_url=f"{self.base_url}/")
        payload = client.healthz()

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["model"], "context-1")
        self.assertEqual(payload["provider"], "unit-test")

    def test_agent_step_posts_payload_and_returns_json(self):
        client = Context1Client(base_url=self.base_url)
        payload = {
            "trajectory": [{"role": "user", "content": "hello"}],
            "tool_metadata": [{"name": "search", "description": "search"}],
        }

        response = client.agent_step(payload)

        self.assertTrue(response["ok"])
        self.assertEqual(response["received"], payload)
        self.assertEqual(_Context1TestHandler.last_request["path"], "/v1/agent/step")
        self.assertEqual(
            _Context1TestHandler.last_request["headers"]["Content-Type"],
            "application/json",
        )

    def test_agent_step_stream_yields_sse_events(self):
        client = Context1Client(base_url=self.base_url)
        response = Mock()
        response.iter_lines.return_value = [
            "event: message",
            'data: {"delta":"hello"}',
            "",
            'data: {"delta":"world"}',
            "data: [DONE]",
        ]

        events = list(client._iter_sse(response))

        self.assertEqual(events, [{"delta": "hello"}, {"delta": "world"}])


if __name__ == "__main__":
    unittest.main()
