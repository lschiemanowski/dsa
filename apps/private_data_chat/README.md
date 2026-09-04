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

`chainlit_app.py` implements one deliberately small Chainlit application:

1. it rebuilds the untrusted model request from bounded user/assistant text, the operator's
   synthetic `MockDatabaseContext`, and an optional digest-pinned database card;
2. the clarifier asks questions until it emits a validated quantitative proposal, optionally with
   untrusted analysis guidance when the host enables it;
3. Chainlit retains the canonical proposal as an ordinary chat message, then shows native
   confirmation actions in a separate transient prompt;
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

## Chainlit setup

The application remains in the separate `apps.private_data_chat` namespace, but the project wheel
includes it so a cloned checkout does not need a custom `PYTHONPATH`. The product-level `dsa`
command is only a dispatcher: `dsa chat` enters this application package, while `dsa run` and
`dsa benchmark` enter their existing core CLI modules. Nothing under `src/dsa` imports the chat
application or Chainlit.

Chainlit is an exact-pinned optional dependency and is not vendored. Install it from the frozen
project lockfile:

```bash
uv sync --python 3.12 --extra chat --frozen
```

`dsa chat` uses the application-owned Chainlit configuration and monochrome DSA mark. The demo
keeps one dark theme, hides the theme control and generic watermark, and disables spontaneous file
uploads and message editing.

For a portable, non-secret starting point, copy and edit the Online Retail II TOML template:

```bash
cp apps/private_data_chat/online-retail-ii.chat.toml.example private-data-chat.toml
```

Paths in this file are resolved relative to the file itself. Its `[database_card]` section pins an
exact bounded JSON card stored alongside the DuckDB in the Hugging Face dataset. The application
verifies its repository, immutable commit, path, and SHA-256 digest before parsing it. The welcome
message shows the card's user-facing inventory while the untrusted clarifier receives the complete
card, including technical analysis notes. Those notes are never rendered in the welcome message.
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

Then launch the application. Chainlit opens the browser by default:

```bash
uv run dsa chat --config private-data-chat.toml
```

For a remote or already-open browser, add `--headless`; `--host`, `--port`, `--watch`, and `--debug`
are also available. Run `uv run dsa chat --help` for the exact interface.

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
the same safe `ModelConfiguration` contract used by DSA validates both. Enable MLflow reporting
only when an MLflow backend is already configured.

Analysis guidance is disabled by default. Set `enable_analysis_guidance = true` in the TOML file,
or `DSA_CHAT_ENABLE_ANALYSIS_GUIDANCE=true` when using environment-only configuration, to let the
clarifier add a short ordered plan with optional SQL or Python snippets to the proposal. The
confirmation view includes the exact guidance and its digest. The application never executes those
snippets directly: after approval, the adapter labels the guidance as untrusted and the trusted
model must inspect the real database, correct or ignore the suggestions, and produce the ordinary
DSA derivation. Only that final derivation is replayed and eligible for the notebook download.
Guidance is also rendered separately as wrapped prose for readability; the canonical JSON remains
beneath it as the exact approved representation.

The Chainlit process needs access to the Docker CLI and daemon used for derivation replay. The
configured database and runs directory must also be visible to that daemon at the same absolute
paths.

`DSA_CHAT_CLARIFIER_MODEL_NAME` selects the capable untrusted model. The application creates it as
a tool-free Pydantic AI agent and sends only bounded user/assistant text plus the synthetic mock
context. `DSA_CHAT_TRUSTED_MODEL_NAME` is resolved separately inside `DsaAnalysisExecutor`; its
request includes the approved question, answer schema, and—only when enabled and explicitly
approved—clearly labeled untrusted analysis guidance. It never receives the clarification
conversation.

The notebook is delivered through Chainlit's native `File` element under a fixed download name.
Before attaching it, the adapter reopens the exact retained file without following symlinks and
rechecks its byte length and SHA-256 digest. No host path is put in the chat.

`clarification-skill.md` is a host-owned system prompt loaded by the application. It improves the
clarifier's behavior; the parsed contracts, confirmation digest, and trusted adapter enforce the
actual boundary.

See [THREAT_MODEL.md](THREAT_MODEL.md) for the boundary assumptions and demo limitations.
