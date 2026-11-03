# Asset Master Match

Azure Function (Python 3.11) that recommends the best Pinecone master-data entry for asset create, using Gemini for selection and optional MakeModel refinement.

## Endpoint

`POST /api/asset_master_match`  
Auth: **Anonymous** (same pattern as validation-ai)

## Quick start (local)

1. Copy settings and fill secrets:

```bash
cp local.settings.json.example local.settings.json
```

2. Create a venv and install deps (Python 3.11) — do this yourself before running anything:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

`requirements-dev.txt` adds **pytest** and **ruff** on top of the runtime deps.

3. Start the function host:

```bash
func start
```

4. Call the endpoint with a sample from `samples/sample_payloads.json`.

## Lint and tests

With the venv active and deps installed:

```bash
ruff check .
ruff format --check .
pytest
```

Auto-fix formatting (optional):

```bash
ruff format .
```

Config lives in `pyproject.toml` (`[tool.ruff]`, `[tool.pytest.ini_options]`).

## Request shape

**Preferred (Option A)** — one or more customers; each group has its own `CustomerId` (Pinecone never mixes customers):

```json
{
  "requests": [
    {
      "CustomerId": 1001,
      "assets": [
        { "AssetName": "Hp Probook 430 Gb", "CategoryId": 1036, "AssetClassId": 1066 }
      ]
    },
    {
      "CustomerId": 1004,
      "assets": [
        { "AssetName": "hdd 16 gb ram", "CategoryId": 1039, "AssetClassId": 1107 }
      ]
    }
  ]
}
```

**Legacy (still supported)** — single customer for the whole batch:

```json
{
  "CustomerId": 1004,
  "assets": [
    {
      "AssetName": "hdd 16 gb ram",
      "CategoryId": 1039,
      "AssetClassId": 1107,
      "Cost": 50000,
      "Count": 1
    }
  ]
}
```

- Prefer `requests[]`: each group’s `CustomerId` is required and used only for that group’s assets.
- `AssetName` (required per asset) is the primary match signal (max **800** chars by default).
- `AssetDescription` (optional) is appended to the embed query and passed to Gemini select when present (truncated; max **800** chars). Final embed query is capped at **1000** chars (name preserved).
- `CategoryId` / `AssetClassId` are optional Pinecone filters (numeric ids only).
- Every asset must be a JSON **object** (non-objects → HTTP 400).
- All other input fields are echoed back unchanged.
- Max **50** assets **total** across all customers; up to **10** processed in parallel (configurable).

## Response shape

Flat `results` in input order. Each row echoes its own `CustomerId` (no top-level `CustomerId`).

```json
{
  "results": [
    {
      "CustomerId": 1004,
      "AssetName": "hdd 16 gb ram",
      "CategoryId": 1039,
      "AssetClassId": 1107,
      "master": {
        "AssetClass": "...",
        "AssetClassId": 1107,
        "Category": "Machinery",
        "CategoryId": 1039,
        "SubCategory": "...",
        "SubCategoryId": 10067,
        "MakeModel": "16 gb ram",
        "MakeModelId": null
      },
      "MatchFound": "Y",
      "MakeModelGenerated": "Y",
      "Message": null
    }
  ]
}
```

| Flag | Meaning |
|------|---------|
| `MatchFound: Y` | At least one Pinecone hit; one candidate chosen |
| `MatchFound: N` | No candidates (or item error); `master` is `null` |
| `MakeModelGenerated: Y` | MakeModel was refined; `MakeModelId` is `null` |
| `MakeModelGenerated: N` | MakeModel taken from Pinecone; id kept |

`Message` is **fixed text from this service only** (never free-form LLM prose):

| `Message` | When |
|-----------|------|
| `null` | Normal successful match |
| `fallback_index` | LLM index invalid; candidate #1 used |
| `No master data candidates found` | Zero Pinecone hits |
| `AssetName is required` / `AssetName exceeds maximum length` | Input validation |
| `Failed to embed AssetName` | Embedding returned empty |
| `Embedding service unavailable` / `Embedding service error` | Embed upstream failure |
| `Master data search unavailable` | Pinecone failure |
| `Gemini selection unavailable` | Gemini select 503 / high demand |
| `Gemini rate limit hit` | Local or upstream select rate/quota limit |
| `Failed to select master match` | Other Gemini select failures (bad JSON, empty reply, etc.) |
| `Match pipeline error` | Unexpected worker failure |

Exception detail stays in server logs / `func` stderr only (never echoed into `Message`).

Per-item failures set `Error: true` and a `Message`; other assets in the batch still return.

## Settings (important)

See `local.settings.json.example`. Never commit `local.settings.json`.

| Key | Role |
|-----|------|
| `GEMINI_API_KEY` / `GEMINI_MODEL_NAME` | LLM (swap 3.1 ↔ 2.5 Flash Lite here) |
| `EMBEDDING_MODEL_NAME` / `EMBEDDING_DIMENSION` | Embeddings (`1536`) |
| `PINECONE_API_KEY` / `PINECONE_INDEX_NAME` | Vector index |
| `PINECONE_TOP_K` | Default `5` |
| `MAX_ASSETS_PER_REQUEST` / `MAX_PARALLEL_ITEMS` | Batch size (50) / parallel workers (10) |
| `GEMINI_MAX_CALLS_PER_MINUTE` | Local select cap (default `120`; hour/day still 500 / 5000) |
| `PINECONE_MAX_CALLS_PER_MINUTE` | Local query cap (default `100`; hour/day still 1000 / 10000) |
| `MAX_ASSET_NAME_LENGTH` / `MAX_ASSET_DESCRIPTION_LENGTH` | Input name / optional description caps |
| `MAX_EMBED_QUERY_LENGTH` | Cap on name(+description) text sent to embed (default `1000`) |
| `MAX_MAKE_MODEL_LENGTH` | Generated MakeModel cap |
| `GEMINI_MAX_INPUT_TOKENS` / `GEMINI_MAX_PROMPT_CHARS` | Prompt size guard before Gemini |
| Timeouts / retries / call caps | Cost and hang protection |

## Samples for the team

Shareable dummy requests: [`samples/sample_payloads.json`](samples/sample_payloads.json)

## Open questions for senior (throughput / production)

Start knobs are set for **50 assets/request** and **10 parallel**. Backend traffic will be continuous — please confirm before raising hour/day or Azure timeouts:

1. Peak HTTP calls/min and assets/min from create-asset?
2. Usual batch size (1–5 vs often 50)?
3. Real Gemini RPM/TPM — do local caps (120/min, 500/hr, 5000/day) match the Google project?
4. Should embeddings count toward Gemini local limits? (Today only select/`generate_json` counts.)
5. Pinecone: free Starter is OK for parallel 10; when do we need paid (monthly RU / SLA)?
6. Is `functionTimeout` 2 min enough for worst-case 50 @ parallel 10, or raise / use Premium?
7. `host.json` `http.maxConcurrentRequests` is 5 — raise if multiple callers?
8. On rate-limit / stage errors: backend retry+backoff or queue?
9. Leave hour/day as-is or set from measured peak?
10. Separate embed / Pinecone / LLM timeouts vs single Gemini timeout?
11. Separate Pinecone concurrency vs shared `MAX_PARALLEL_ITEMS`?
12. Prod `GEMINI_MODEL_NAME` (3.1 vs 2.5 flash lite)?
13. Auth: Anonymous behind gateway vs Function key / APIM?

## Docs for agents

`cursor.md` — product rules, stack, and changelog (update on every change; gitignored).
