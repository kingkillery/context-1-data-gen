# Chroma Context-1 Data Generation

Synthetic data generation pipeline from the [Context-1 technical report](https://www.trychroma.com/research/context-1).

This repository now covers two adjacent workflows:

- local synthetic-data generation across the existing web, SEC, patents, and email domains
- remote serving of `chromadb/context-1` as a hosted search subagent that local harnesses can call

Context-1 model weights are available at [Hugging Face](https://huggingface.co/chromadb/context-1).

## Setup

```bash
# Install dependencies
uv sync

# For all optional dependencies (reranking, patents, indexing)
uv sync --all-extras

# Configure environment
cp .env.example .env
```

## Environment

| Variable | Used by |
|----------|---------|
| `MINIMAX_API_KEY` | Default LLM provider for all generation stages |
| `MINIMAX_BASE_URL` | Anthropic-compatible MiniMax endpoint |
| `ANTHROPIC_API_KEY` | Optional fallback provider when MiniMax is not configured |
| `DEFAULT_LLM_MODEL` | Default generation model, defaults to `minimax-m2.7-highspeed` |
| `DEFAULT_VERIFY_MODEL` | Default verification model, defaults to `minimax-m2.7-highspeed` |
| `SERPER_API_KEY` | Web search + scraping |
| `JINA_API_KEY` | Web backup page fetcher |
| `OPENAI_API_KEY` | Embeddings for web, SEC, email, and patents |
| `CHROMA_API_KEY` | Chroma indexing for web, SEC, email, and patents |
| `CHROMA_DATABASE` | Chroma indexing target database |
| `BASETEN_API_KEY` | SEC reranking |
| `CONTEXT1_BASE_URL` | Hosted Context-1 base URL, defaults to `https://context1.pkking.computer` |
| `CONTEXT1_HOSTNAME` | Public Context-1 hostname, defaults to `context1.pkking.computer` |
| `FRONTIER_WS_URL` | Local Codex appserver websocket, defaults to `ws://127.0.0.1:4500` |

### Provider behavior

- Local scripts still call `get_anthropic_client()`, but the helper is now provider-aware.
- If `MINIMAX_API_KEY` is present, the Anthropic SDK is instantiated against `MINIMAX_BASE_URL`.
- If MiniMax is not configured, the code falls back to `ANTHROPIC_API_KEY`.
- The default model for local generation stages is `minimax-m2.7-highspeed`.
- OpenAI remains the embedding provider.

## Hosted Context-1

The intended runtime split is:

- local scripts, corpora, and project-search tools stay local
- `chromadb/context-1` is served remotely as an inference-only search subagent
- frontier reasoning stays local through `FRONTIER_WS_URL`

### API contract

- `GET /healthz` returns readiness, model name, and provider details
- `POST /v1/agent/step` accepts a serialized trajectory, tool metadata, and generation config
- streaming responses are exposed as server-sent events so local harnesses can render incremental agent output

### Colab notebook

Use [`notebooks/context1_colab_server.ipynb`](notebooks/context1_colab_server.ipynb) to:

- install `vllm`, `fastapi`, `uvicorn`, and `cloudflared`
- authenticate to Hugging Face
- launch `chromadb/context-1` behind a local OpenAI-compatible vLLM server
- expose a lightweight agent API at `/v1/agent/step`
- connect Colab to Cloudflare using `CLOUDFLARE_TUNNEL_TOKEN`

Expected notebook secrets:

- `HF_TOKEN`
- `CLOUDFLARE_TUNNEL_TOKEN`
- optional `MODEL_NAME`, `PUBLIC_HOSTNAME`, `VLLM_PORT`, and `API_PORT`

## Cloudflare front door

The Worker implementation lives in [`cloudflare/context1_frontdoor/src/index.js`](cloudflare/context1_frontdoor/src/index.js).

Recommended deployment shape:

- `context1.pkking.computer` is the stable public hostname
- the notebook publishes a tunnel-backed origin such as `context1-origin.pkking.computer`
- the Worker checks the origin health endpoint before proxying
- when the origin is down:
  - browser `GET /` returns a clean offline page
  - API routes return structured `503` JSON instead of a generic proxy or tunnel error

The included [`wrangler.toml`](cloudflare/context1_frontdoor/wrangler.toml) defaults `ORIGIN_BASE_URL` to `https://context1-origin.pkking.computer`. Adjust that if your origin hostname differs.

## Domains

Each domain has its own pipeline command and README:

- [Web](agentic_search_data_gen/domains/web/README.md)
- [SEC](agentic_search_data_gen/domains/sec/README.md)
- [Patents](agentic_search_data_gen/domains/patents/README.md)
- [Email (Epstein)](agentic_search_data_gen/domains/epstein/README.md)

## Project structure

```text
agentic_search_data_gen/
|-- core/
|   |-- context1_client.py   # Hosted Context-1 REST client
|   |-- explore.py           # BaseExplorerAgent
|   |-- extend.py            # BaseExtenderAgent
|   |-- verify.py            # BaseVerifier
|   |-- distract.py          # BaseDistractorAgent
|   |-- rerank.py            # Baseten reranker client
|   |-- indexing.py          # Chroma indexing utilities
|   `-- utils.py             # Provider-aware client config and shared helpers
`-- domains/
    |-- web/
    |-- sec/
    |-- patents/
    `-- epstein/
```
