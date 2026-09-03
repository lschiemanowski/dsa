# Private Data Chat

This directory contains an application built on top of DSA. It is deliberately outside
`src/dsa`: none of its UI, approval, identity, or deployment policy belongs to the reusable
analysis library.

The application lets a user refine a question with a capable but untrusted model using only a
small synthetic database description. After the user approves the exact quantitative question,
a trusted broker runs DSA against the real database with a separately configured trusted model.
The user receives the structured answer and, when DSA successfully validates a derivation, a
download control for the generated notebook. That result ends the conversation: another user
message is refused before either model is called.

## Implemented demo flow

`chainlit_app.py` implements one deliberately small Chainlit application:

1. it rebuilds the untrusted model request from bounded user/assistant text and the operator's
   synthetic `MockDatabaseContext` only;
2. the clarifier asks questions until it emits a validated quantitative proposal;
3. Chainlit displays the canonical proposal with native confirmation actions;
4. exact confirmation invokes the host-owned `DsaAnalysisExecutor` against the real database;
5. the application returns the structured answer and attaches a notebook download when DSA
   retained a verified derivation; and
6. that assistant result marks the conversation terminal. Later messages tell the user to open a
   new conversation without calling the clarifier or DSA.

The contracts supporting that flow are intentionally modest:

- `ProposalPayload` is the only model-authored proposal content.
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
includes it so a cloned checkout does not need a custom `PYTHONPATH`. Chainlit is an exact-pinned
optional dependency and is not vendored. From the repository root, one command resolves the
optional UI environment and starts the application:

```bash
uv run --python 3.12 --extra chat --frozen \
  chainlit run apps/private_data_chat/chainlit_app.py
```

Before starting it, configure the host-owned boundary:

```bash
export DSA_CHAT_DATA_SOURCE_ID=retail
export DSA_CHAT_MOCK_CONTEXT_PATH="$PWD/apps/private_data_chat/mock-database.example.json"
export DSA_CHAT_CLARIFIER_MODEL_NAME=openai:your-capable-untrusted-model
export DSA_CHAT_CLARIFIER_MODEL_SETTINGS_JSON='{}'
export DSA_CHAT_DATABASE_PATH=/absolute/path/to/real.duckdb
export DSA_CHAT_RUNS_DIRECTORY=/absolute/path/to/dsa-runs
export DSA_CHAT_TRUSTED_MODEL_NAME=openai:your-trusted-model
export DSA_CHAT_TRUSTED_MODEL_SETTINGS_JSON='{}'
export DSA_CHAT_DOCKER_IMAGE='repository/image@sha256:...'
export DSA_CHAT_REPORT_TO_MLFLOW=false
```

Provider credentials continue to come from the process environment. Neither model-settings
variable may contain credentials or a provider endpoint; the same safe `ModelConfiguration`
contract used by DSA validates both. Set `DSA_CHAT_REPORT_TO_MLFLOW=true` only when an MLflow
backend is already configured.

The Chainlit process needs access to the Docker CLI and daemon used for derivation replay. The
configured database and runs directory must also be visible to that daemon at the same absolute
paths.

`DSA_CHAT_CLARIFIER_MODEL_NAME` selects the capable untrusted model. The application creates it as
a tool-free Pydantic AI agent and sends only bounded user/assistant text plus the synthetic mock
context. `DSA_CHAT_TRUSTED_MODEL_NAME` is resolved separately inside `DsaAnalysisExecutor`; its
request includes the approved question and answer schema, not the clarification conversation.

The notebook is delivered through Chainlit's native `File` element under a fixed download name.
Before attaching it, the adapter reopens the exact retained file without following symlinks and
rechecks its byte length and SHA-256 digest. No host path is put in the chat.

`clarification-skill.md` is a host-owned system prompt loaded by the application. It improves the
clarifier's behavior; the parsed contracts, confirmation digest, and trusted adapter enforce the
actual boundary.

See [THREAT_MODEL.md](THREAT_MODEL.md) for the boundary assumptions and demo limitations.
