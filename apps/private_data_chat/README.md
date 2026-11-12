# Private Data Chat

This directory contains an application built on top of DSA. It is deliberately outside
`src/dsa`: none of its UI, approval, identity, or deployment policy belongs to the reusable
analysis library.

The application lets a user refine a question with a capable but untrusted model using only an
operator-approved database card and synthetic example rows. An opt-in mode also lets
that model suggest concise analysis guidance, including code fragments, as part of the proposal.
After the user approves the exact quantitative question and any guidance, a trusted broker runs
DSA against the real database with a separately configured trusted model. The user receives the
structured answer and, when DSA successfully validates a derivation, a download control for the
generated notebook. That result ends the conversation: another user message is refused before
either model is called.

## Implemented demo flow

The chat uses an assistant-ui frontend and a FastAPI server:

1. it rebuilds the untrusted model request from bounded user/assistant text, the operator's
   synthetic `MockDatabaseContext`, and an optional digest-pinned database card;
2. the clarifier asks questions until it emits a validated quantitative proposal, optionally with
   untrusted analysis guidance when the host enables it;
3. The chat displays the proposal with its scope and output fields, an expandable view of the
   exact request JSON, and controls to approve or refine it;
4. exact confirmation invokes the host-owned `DsaAnalysisExecutor` against the real database;
5. the application returns the structured answer and attaches a notebook download when DSA
   retained a verified derivation; and
6. that assistant result marks the conversation terminal. Later messages tell the user to open a
   new conversation without calling the clarifier or DSA.

The contracts supporting that flow are intentionally modest:

- `ProposalPayload` is the only model-authored proposal content. Its optional
  `analysis_guidance` is bounded and is treated as unexecuted advice, not a derivation receipt.
- `ProposalBinding` is host-authored and binds the proposal to a user, conversation, and logical
  data source.
- the user approves the digest of both objects, not mutable chat text;
- `PrivateDataBroker` enforces the proposal state machine and at-most-once execution;
- `AnalysisRequest` contains a logical data-source ID, never a database path, model credentials,
  provider endpoint, Docker configuration, or run policy;
- `AnalysisResult` exposes opaque artifact identities, never host paths or raw exceptions; and
- the answer schema uses a deliberately small, bounded JSON Schema subset suitable for a first
  public boundary.

The in-memory store is sufficient because proposal, confirmation, and execution happen in one
application session. The session atomically rejects overlapping turns for the same conversation;
it releases that reservation after clarification or declined confirmation and makes it permanently
terminal immediately after confirmation. This is a portfolio demo, not durable workflow
infrastructure.

## State machine

```text
proposed --approve exact digest--> approved --execute once--> running
    |                                                   /          \
    +--------------------expiry----------------> expired   succeeded  failed
```

Approval and execution require the same user and conversation that created the proposal. An
identity mismatch is reported as `proposal_not_found` to avoid exposing another user's records.
Repeated approval is idempotent. Repeated execution returns the retained result after completion;
it never starts a second analysis. A concurrent call while execution is in progress receives
`proposal_execution_in_progress`.

The single-process lock makes those statements true for `InMemoryProposalStore` during the one
invocation.

## Install and launch

The wheel includes the compiled assistant-ui frontend. Install the wheel with the `chat` extra,
then use `dsa chat`. The installed app runs with Python and opens in a browser.

```bash
uv pip install '/path/to/dsa-0.1.0-py3-none-any.whl[chat]'
```

From a source checkout, install the locked Python dependencies and build the frontend once with
Node.js 22.12 or newer:

```bash
uv sync --python 3.12 --extra chat --frozen
npm --prefix ui ci
npm --prefix ui run build
```

`uv build` compiles the frontend automatically and includes it in the wheel.

For a portable, non-secret starting point, copy and edit the Online Retail II TOML template:

```bash
cp apps/private_data_chat/online-retail-ii.chat.toml.example private-data-chat.toml
```

Paths in this file are resolved relative to the file itself. Its `[database_card]` and
`[synthetic_context]` sections pin the description and explicitly fake rows stored alongside
the DuckDB on Hugging Face. Both pin the same repository revision, with separate content
digests. The application verifies bounded downloads and checks that the represented
relations and columns agree before any model call. The welcome
message shows the card's user-facing inventory while the untrusted clarifier receives the complete
card, including technical analysis notes. Card fields are escaped as literal text before Markdown
rendering, and those notes are never rendered in the welcome message.
For offline use, set `mock_context_path` to a local synthetic-context JSON file
instead of configuring `[synthetic_context]`.
Its `[clarifier]` and `[trusted]` sections are parsed
independently through the safe model-configuration contract; credentials and provider endpoints
are rejected there and must remain in environment variables. For example, a remote OpenRouter
clarifier and a trusted local llama-server use:

```bash
export OPENROUTER_API_KEY=<token>
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
```

Check the optional dependencies, typed configuration, synthetic context, and pinned public
database card without starting a server or calling either model:

```bash
uv run dsa chat --config private-data-chat.toml --check
```

Then launch the application. The launcher opens the browser by default:

```bash
uv run dsa chat --config private-data-chat.toml
```

Use `--headless` to open the browser yourself. The default URL is `http://127.0.0.1:8001/`.
Use `--host` and `--port` to set the listen address, `--watch` to reload after Python edits,
and `--debug` for verbose server logs. The API accepts loopback browser addresses.

The previous environment-only configuration remains supported. Source and edit
`online-retail-ii.env.example`, then run `uv run dsa chat` without `--config`.

The database setting must name a regular verified database file. Hugging Face snapshot entries are
normally symlinks and are intentionally rejected by the DSA run boundary; use the verified blob
path returned by `load_huggingface_evaluation_pack`, or copy those verified bytes to a private
regular file. The template assumes a local OpenAI-compatible model server on port 8000 and a
remote OpenRouter clarifier. Export `OPENROUTER_API_KEY` separately rather than saving it in the
template.

Provider credentials continue to come from the process environment. Neither model-settings
section nor its environment-variable equivalent may contain credentials or a provider endpoint;
the same safe `ModelConfiguration` contract used by DSA validates both.

Logfire is off by default. To enable it for the trusted analysis, install the `logfire` extra,
set `LOGFIRE_TOKEN` in the host environment, and set `report_to_logfire = true` in the TOML
or `DSA_CHAT_REPORT_TO_LOGFIRE=true` when using environment-only configuration.
Logfire receives operational metadata such as timings, usage, and failure codes. Prompts,
answers, tool arguments and results, and retained files stay out of telemetry.
Existing MLflow settings set to false can stay in place. Remove any enabled MLflow setting
and enable Logfire explicitly if you want telemetry.

Analysis guidance is disabled by default. Set `enable_analysis_guidance = true` in the TOML file,
or `DSA_CHAT_ENABLE_ANALYSIS_GUIDANCE=true` when using environment-only configuration, to let the
clarifier add a short ordered plan with optional SQL or Python snippets to the proposal. The
confirmation view includes the exact guidance and its digest. The application never executes those
snippets directly: after approval, the adapter labels the guidance as untrusted and the trusted
model must inspect the real database, correct or ignore the suggestions, and produce the ordinary
DSA derivation. Only that final derivation is replayed and eligible for the notebook download.
Guidance is also rendered separately as wrapped literal text for readability, with Markdown
punctuation neutralized; the canonical JSON remains beneath it as the exact approved
representation.

The chat server needs access to the Docker CLI and daemon used for derivation replay. The
configured database and runs directory must also be visible to that daemon at the same absolute
paths.

`DSA_CHAT_CLARIFIER_MODEL_NAME` selects the capable untrusted model. The application creates it as
a tool-free Pydantic AI agent and sends only bounded user/assistant text plus the synthetic mock
context. `DSA_CHAT_TRUSTED_MODEL_NAME` is resolved separately inside `DsaAnalysisExecutor`; its
request includes the approved question, answer schema, and—only when enabled and explicitly
approved—clearly labeled untrusted analysis guidance. It never receives the clarification
conversation.

Use **Open notebook** to view and download the verified derivation. For each request, the server
reopens the retained file without following symlinks and checks its byte length and SHA-256 digest.

`clarification-skill.md` is a host-owned system prompt loaded by the application. It improves the
clarifier's behavior; the parsed contracts, confirmation digest, and trusted adapter enforce the
actual boundary.

See [THREAT_MODEL.md](THREAT_MODEL.md) for the boundary assumptions and demo limitations.

## HTTP API development

Start the single-process API on loopback with the same host configuration as the chat launcher:

```bash
DSA_CHAT_CONFIG_PATH="$PWD/private-data-chat.toml" \
  uv run --extra chat uvicorn apps.private_data_chat.http_api:create_app --factory \
  --host 127.0.0.1 --port 8001
```

Use the interactive API docs at `http://127.0.0.1:8001/docs`. The API uses an HTTP-only cookie to
identify the owning browser session. Keep the same cookie and conversation ID across requests.

| Method and path | Request and result |
| --- | --- |
| `POST /api/conversations` | Create a conversation and set the session cookie. |
| `POST /api/conversations/{id}/messages` | Send `{"text": "..."}` and get the updated conversation. |
| `POST /api/conversations/{id}/proposals/{proposal_id}/decision` | Send `{"decision": "approve", "digest": "..."}` or use `"revise"`. Approval starts one run and returns HTTP 202. |
| `GET /api/conversations/{id}` | Get the transcript, pending proposal ID, and run state. Poll while the status is `running`. |
| `GET /api/conversations/{id}/notebook` | Download the verified notebook when the result includes a notebook URL. |

Proposal messages contain a structured `payload` and server-serialized `request_json`. A successful
result includes the structured `answer` and `response_json`. The serialized forms preserve the exact
values for inspection. Review uses the stored proposal ID and digest, and repeated approvals
return the existing run state. An approval accepted before expiry claims its run before the
executor starts.

The HTTP API keeps up to eight conversations per browser and 64 per process. Conversations are
removed after one hour of inactivity. Clarification and analysis in progress are kept until they
finish. Up to two analyses can run at a time. If both slots are occupied, approval returns HTTP
429 with `analysis_capacity_reached` and the proposal can be approved again while it is valid.
Sessions are held in memory and are lost on server restart. The API accepts localhost and IPv4
and IPv6 loopback.

## Develop the assistant-ui frontend

Start the FastAPI server from the repository root using the command above. In a second terminal
with Node.js 22.12 or newer, run:

```bash
cd ui
npm ci
npm run dev
```

Open the loopback URL printed by Vite. Vite proxies `/api` to the Python server on port 8001.
Set `DSA_CHAT_API_ORIGIN` before starting Vite if the API is listening on another loopback port.

The frontend keeps the current conversation ID in tab session storage so a reload can recover
the transcript and a running analysis. The proposal and result cards offer their exact JSON in
disclosures. **Open notebook** fetches the verified derivation and shows its saved cells and
outputs alongside the chat, or in a full-width pane on narrow screens.

Run `npm run build` to type-check and build the UI. The browser smoke test starts Vite and the
real HTTP API with a deterministic clarifier and executor. From `ui/`, after installing the
Python `chat` extra and npm dependencies, run:

```bash
npx playwright install chromium
npm run test:browser
```

Set `DSA_TEST_BUILD=1` to exercise the production bundle with the same smoke test, or
`DSA_TEST_CHROMIUM` to use a compatible Chromium executable already installed locally.

## Optional plots

Ask for a chart and review `allow_plots: true` in the proposal before approving.
The trusted analysis may return up to three static PNGs alongside the answer, with
downloads and embedded outputs in the verified notebook. Plots never return to the
untrusted clarifier.
