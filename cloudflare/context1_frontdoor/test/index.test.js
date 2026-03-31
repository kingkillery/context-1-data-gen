import assert from "node:assert/strict";
import { test } from "node:test";

import handler from "../src/index.js";

async function withMockedFetch(mockFetch, run) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = mockFetch;
  try {
    await run();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

function toRequest(input, init) {
  if (input instanceof Request) {
    return input;
  }

  const url = input instanceof URL ? input.href : input;
  return new Request(url, init);
}

test("returns an offline HTML page when the origin is down", async () => {
  await withMockedFetch(async (input, init) => {
    const request = toRequest(input, init);
    assert.equal(new URL(request.url).pathname, "/healthz");
    throw new Error("origin unavailable");
  }, async () => {
    const response = await handler.fetch(
      new Request("https://context1.pkking.computer/", { method: "GET" }),
      {
        ORIGIN_BASE_URL: "https://context1-origin.example",
        SERVICE_HOSTNAME: "context1.pkking.computer",
      },
    );

    assert.equal(response.status, 200);
    assert.equal(response.headers.get("content-type"), "text/html; charset=utf-8");
    assert.equal(response.headers.get("cache-control"), "no-store");
    const body = await response.text();
    assert.match(body, /Temporarily offline/);
    assert.match(body, /context1\.pkking\.computer/);
  });
});

test("returns structured 503 JSON for API routes when the origin is down", async () => {
  await withMockedFetch(async () => {
    throw new Error("origin unavailable");
  }, async () => {
    const response = await handler.fetch(
      new Request("https://context1.pkking.computer/v1/agent/step", { method: "POST" }),
      {
        ORIGIN_BASE_URL: "https://context1-origin.example",
        SERVICE_HOSTNAME: "context1.pkking.computer",
      },
    );

    assert.equal(response.status, 503);
    assert.equal(response.headers.get("cache-control"), "no-store");
    const body = await response.json();
    assert.equal(body.error.type, "service_unavailable");
    assert.match(body.error.message, /Reconnect the Colab notebook/);
  });
});

test("serves the online status document for the root path when the origin is healthy", async () => {
  let healthChecks = 0;
  await withMockedFetch(async (input, init) => {
    const request = toRequest(input, init);
    healthChecks += 1;
    assert.equal(new URL(request.url).pathname, "/healthz");
    return new Response("ok", { status: 200 });
  }, async () => {
    const response = await handler.fetch(
      new Request("https://context1.pkking.computer/", { method: "GET" }),
      {
        ORIGIN_BASE_URL: "https://context1-origin.example",
        SERVICE_HOSTNAME: "context1.pkking.computer",
      },
    );

    assert.equal(healthChecks, 1);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
    const body = await response.json();
    assert.equal(body.service, "context-1");
    assert.equal(body.status, "online");
    assert.deepEqual(body.endpoints, ["/healthz", "/v1/agent/step"]);
  });
});

test("proxies API routes to the origin when the origin is healthy", async () => {
  const seen = [];
  await withMockedFetch(async (input, init) => {
    const request = toRequest(input, init);
    const pathname = new URL(request.url).pathname;
    seen.push(pathname);
    if (pathname === "/healthz") {
      return new Response("ok", { status: 200 });
    }

    const body = await request.text();
    return Response.json(
      {
        proxied: true,
        method: request.method,
        contentType: request.headers.get("content-type"),
        body,
      },
      { status: 200 },
    );
  }, async () => {
    const response = await handler.fetch(
      new Request("https://context1.pkking.computer/v1/agent/step", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt: "hello" }),
      }),
      {
        ORIGIN_BASE_URL: "https://context1-origin.example",
        SERVICE_HOSTNAME: "context1.pkking.computer",
      },
    );

    assert.deepEqual(seen, ["/healthz", "/v1/agent/step"]);
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.proxied, true);
    assert.equal(body.method, "POST");
    assert.equal(body.contentType, "application/json");
    assert.equal(body.body, "{\"prompt\":\"hello\"}");
  });
});
