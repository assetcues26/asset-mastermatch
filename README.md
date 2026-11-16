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

4. `POST` to `http://localhost:7071/api/asset_master_match` with a JSON body from the **Request shape** examples below (e.g. Postman).

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

**Preferred (Option A)** — nested `requests[]`; one or more customers; each group has its own `CustomerId` (Pinecone never mixes customers):

```json
{
  "requests": [
    {
      "CustomerId": 1001,
      "assets": [
        {
          "AssetName": "Hp Probook 430 Gb",
          "CategoryId": 1036,
          "AssetClassId": 1066
        }
      ]
    },
    {
      "CustomerId": 1299,
      "assets": [
        {
          "AcquisitionDate": "2024-01-15",
          "AssetNumber": "",
          "SubNumber": 0,
          "AssetName": "Metal Almirah",
          "AssetType": "PP&E",
          "PlantLocation": "Vedanta Aluminium - Admin Block",
          "Cost": 14000,
          "Count": 2
        },
        {
          "AcquisitionDate": "2024-03-01",
          "AssetNumber": "",
          "SubNumber": 0,
          "AssetName": "Voltas cassette AC 2.0 ton inverter",
          "AssetDescription": "2.0 ton inverter cassette AC for office block",
          "AssetType": "PP&E",
          "PlantLocation": "Vedanta Aluminium - Office Block A",
          "Cost": 75000,
          "Count": 2
        }
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

### Field rules

- Prefer `requests[]`: each group’s `CustomerId` is required and used only for that group’s assets.
- `AssetName` (required per asset) is the primary match signal (max **800** chars by default).
- `AssetDescription` (optional) is appended to the embed query and passed to Gemini select when present. Overlong description is **truncated** (not a hard error); max **800** chars. Final embed query is capped at **1000** chars (`AssetName` preserved).
- `CategoryId` / `AssetClassId` are optional Pinecone filters (numeric ids only).
- Every asset must be a JSON **object** (non-objects → HTTP 400).
- Max **50** assets **total** across all customers; up to **10** processed in parallel (configurable).

### Echoed input fields

Known keys are copied onto each result row when present:

`AcquisitionDate`, `AssetNumber`, `SubNumber`, `AssetName`, `AssetDescription`, `AssetType`, `PlantLocation`, `AssetClass`, `AssetClassId`, `Category`, `CategoryId`, `Cost`, `Count`

Any other caller fields on the asset object are also echoed (except duplicating top-level `CustomerId` from the group). `CustomerId` on each result comes from the request group / legacy body.

## Response shape

### Success (HTTP 200)

Flat `results` in input order. The body also includes `request_id` (the same id written to the logs). Each row echoes its own `CustomerId` (no top-level `CustomerId`). Nested `master` holds the chosen taxonomy + MakeModel.

```json
{
  "results": [
    {
      "CustomerId": 1299,
      "AcquisitionDate": "2024-01-15",
      "AssetNumber": "",
      "SubNumber": 0,
      "AssetName": "Metal Almirah",
      "AssetType": "PP&E",
      "PlantLocation": "Vedanta Aluminium - Admin Block",
      "Cost": 14000,
      "Count": 2,
      "master": {
        "AssetClass": "Furniture And Fittings",
        "AssetClassId": 5673,
        "Category": "Furniture And Fittings",
        "CategoryId": 2958,
        "SubCategory": "Almirah",
        "SubCategoryId": 83195,
        "MakeModel": "Metal Almirah",
        "MakeModelId": 153356
      },
      "MatchFound": "Y",
      "MakeModelGenerated": "N",
      "Message": null
    },
    {
      "CustomerId": 1299,
      "AcquisitionDate": "2024-03-01",
      "AssetNumber": "",
      "SubNumber": 0,
      "AssetName": "Voltas cassette AC 2.0 ton inverter",
      "AssetDescription": "2.0 ton inverter cassette AC for office block",
      "AssetType": "PP&E",
      "PlantLocation": "Vedanta Aluminium - Office Block A",
      "Cost": 75000,
      "Count": 2,
      "master": {
        "AssetClass": "Office Equipment",
        "AssetClassId": 5669,
        "Category": "Office Equipment",
        "CategoryId": 2744,
        "SubCategory": "Cassette AC",
        "SubCategoryId": 83160,
        "MakeModel": "Voltas 2.0 Ton Inverter Cassette AC",
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
| `Gemini rate limit hit` | Upstream select rate or quota limit (429) |
| `Failed to select master match` | Other Gemini select failures (bad JSON, empty reply, etc.) |
| `Match pipeline error` | Unexpected worker failure |
| `Match timed out` | The batch deadline passed before this asset finished |

Exception detail stays in server logs / `func` stderr only (never echoed into `Message`).

Per-item failures set `Error: true` and a `Message`. `Error`, `Message`, `master`, `MatchFound`, and `MakeModelGenerated` on the input asset are ignored. A batch that still has any row that is not an upstream failure returns HTTP 200. When every row failed because embed, Pinecone, or Gemini failed, or the batch deadline was hit, the status is **503**. Both include `request_id`. A batch of validation errors or genuine no-hit rows stays HTTP 200.

### Batch validation errors (HTTP 400)

Malformed body / request shape failures return before matching:

```json
{ "error": "CustomerId is required (or send requests[])" }
```

Other examples: `requests must be a non-empty array`, `requests[0].assets must be a non-empty array`, `assets[0] must be an object`, `At most 50 assets allowed per request`, `Request body must be JSON`.

## Settings (important)

See `local.settings.json.example`. Never commit `local.settings.json`.

| Key | Role |
|-----|------|
| `GEMINI_API_KEY` / `GEMINI_MODEL_NAME` | LLM (swap 3.1 ↔ 2.5 Flash Lite here) |
| `GEMINI_CALL_TIMEOUT_SEC` | Select timeout per attempt (default `30`). Timeouts and 503/504 are retried once, after 1s. 429 is not retried. |
| `GEMINI_MAX_INPUT_TOKENS` / `GEMINI_MAX_OUTPUT_TOKENS` / `GEMINI_DEFAULT_OUTPUT_TOKENS` | Prompt / generation token caps |
| `GEMINI_MAX_PROMPT_CHARS` | Optional override of prompt char budget (else ~4× input tokens) |
| `EMBEDDING_MODEL_NAME` / `EMBEDDING_DIMENSION` | Embeddings (`1536`) |
| `EMBED_CALL_TIMEOUT_SEC` | Embed timeout (default `15`). Not retried. |
| `PINECONE_API_KEY` / `PINECONE_INDEX_NAME` | Vector index |
| `PINECONE_TOP_K` | Default `5` |
| `PINECONE_CALL_TIMEOUT_SEC` | Query timeout in seconds (default `10`) |
| `BATCH_DEADLINE_SEC` | Stop the request and return finished rows (default `90`). Unfinished rows get `Match timed out`. `functionTimeout` is 4 minutes so this deadline wins. |
| `MAX_ASSETS_PER_REQUEST` / `MAX_PARALLEL_ITEMS` | Batch size (50) / parallel workers (10) |
| `MAX_ASSET_NAME_LENGTH` / `MAX_ASSET_DESCRIPTION_LENGTH` | Input name / optional description caps |
| `MAX_EMBED_QUERY_LENGTH` | Cap on name(+description) text sent to embed (default `1000`) |
| `MAX_MAKE_MODEL_LENGTH` | Generated MakeModel cap (default `150`) |
| `MAX_PROMPT_FIELD_LENGTH` | Per-field truncate in prompts (default `200`) |
| `PROMPT_HOT_RELOAD` | Reload prompt markdown without restart (`false` by default) |

## Open questions for senior (throughput / production)

Start knobs are set for **50 assets/request**, **10 parallel**, and a **90s** batch deadline.

1. Peak HTTP calls/min and assets/min from create-asset?
2. Usual batch size (1–5 vs often 50)?
3. Real Gemini RPM/TPM for this Google project?
4. Pinecone: free Starter is OK for parallel 10; when do we need paid (monthly RU / SLA)?
5. `host.json` `http.maxConcurrentRequests` is 5 — raise if multiple callers?
6. On rate-limit / stage errors: backend retry+backoff or queue?
7. Separate Pinecone concurrency vs shared `MAX_PARALLEL_ITEMS`?
8. Prod `GEMINI_MODEL_NAME` (3.1 vs 2.5 flash lite)?
9. Auth: Anonymous behind gateway vs Function key / APIM?

## Docs for agents

`cursor.md` — product rules, stack, and changelog (update on every change; gitignored).
