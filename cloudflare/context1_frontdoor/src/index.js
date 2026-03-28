const HEALTH_PATH = "/healthz";
const API_PREFIX = "/v1/";

function serviceUnavailableJson(hostname) {
  return Response.json(
    {
      error: {
        type: "service_unavailable",
        message: `The Context-1 service at ${hostname} is offline right now. Reconnect the Colab notebook to restore API access.`,
      },
    },
    { status: 503, headers: { "cache-control": "no-store" } },
  );
}

function offlineHtml(hostname) {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Context-1 Offline</title>
    <style>
      :root {
        color-scheme: dark;
        --bg: #0d1117;
        --panel: rgba(255,255,255,0.05);
        --line: rgba(255,255,255,0.12);
        --text: #e6edf3;
        --muted: #9da7b3;
        --accent: #6ee7b7;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at top, rgba(110,231,183,0.15), transparent 30%),
          linear-gradient(180deg, #0b0f14 0%, var(--bg) 100%);
        color: var(--text);
      }
      main {
        width: min(720px, calc(100vw - 32px));
        padding: 32px;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: var(--panel);
        backdrop-filter: blur(14px);
        box-shadow: 0 24px 80px rgba(0,0,0,0.4);
      }
      .eyebrow {
        font-size: 12px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 12px;
      }
      h1 {
        margin: 0 0 12px;
        font-size: clamp(32px, 6vw, 52px);
        line-height: 0.95;
      }
      p {
        margin: 0 0 12px;
        font-size: 16px;
        line-height: 1.6;
        color: var(--muted);
      }
      code {
        color: var(--text);
        background: rgba(255,255,255,0.08);
        padding: 2px 6px;
        border-radius: 6px;
      }
    </style>
  </head>
  <body>
    <main>
      <div class="eyebrow">Context-1 Search Subagent</div>
      <h1>Temporarily offline</h1>
      <p>The public endpoint at <code>${hostname}</code> is healthy only while the Colab-hosted runtime is connected.</p>
      <p>The model server is currently unavailable. Start or reconnect the Colab notebook and this domain will resume proxying live Context-1 inference traffic.</p>
      <p>API clients should retry once the notebook reports a healthy tunnel.</p>
    </main>
  </body>
</html>`;
}

async function originHealthy(env) {
  try {
    const response = await fetch(new URL(HEALTH_PATH, env.ORIGIN_BASE_URL), {
      method: "GET",
      cf: { cacheTtl: 0, cacheEverything: false },
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function proxyToOrigin(request, env) {
  const url = new URL(request.url);
  const upstream = new URL(url.pathname + url.search, env.ORIGIN_BASE_URL);
  return fetch(new Request(upstream, request));
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const hostname = env.SERVICE_HOSTNAME || url.hostname;
    const healthy = await originHealthy(env);
    const isApiRoute = url.pathname.startsWith(API_PREFIX) || url.pathname === HEALTH_PATH;

    if (!healthy) {
      if (isApiRoute) {
        return serviceUnavailableJson(hostname);
      }
      return new Response(offlineHtml(hostname), {
        status: 200,
        headers: {
          "content-type": "text/html; charset=utf-8",
          "cache-control": "no-store",
        },
      });
    }

    if (!isApiRoute && url.pathname === "/") {
      return new Response(
        JSON.stringify({
          service: "context-1",
          status: "online",
          hostname,
          endpoints: ["/healthz", "/v1/agent/step"],
        }, null, 2),
        {
          headers: {
            "content-type": "application/json; charset=utf-8",
            "cache-control": "no-store",
          },
        },
      );
    }

    return proxyToOrigin(request, env);
  },
};
