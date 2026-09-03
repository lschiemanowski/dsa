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

`openwebui_pipe.py` implements one deliberately small Open WebUI Pipe:

1. it rebuilds the untrusted model request from bounded user/assistant text and the operator's
   synthetic `MockDatabaseContext` only;
2. the clarifier asks questions until it emits a validated quantitative proposal;
3. Open WebUI displays the canonical proposal in a native confirmation dialog;
4. exact confirmation invokes the host-owned `DsaAnalysisExecutor` against the real database;
5. the Pipe returns the structured answer and emits an embedded notebook download when DSA
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
Pipe invocation. The Pipe atomically rejects overlapping turns for the same user and conversation;
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

## Open WebUI setup

This application is intentionally not part of the `dsa` wheel. The Open WebUI backend must have
the repository root on `PYTHONPATH`, the DSA environment available, and access to the Docker CLI
and daemon used for derivation replay. If Open WebUI itself runs in Docker, mount the repository,
the real database, and the runs directory at the same absolute paths seen by the Docker daemon.

In Open WebUI:

1. create a Function using the contents of `openwebui-function.py`;
2. configure its Valves:
   - `CLARIFIER_MODEL_ID`: the capable untrusted model available in Open WebUI. Use a plain model
     connection with no server-side tools, Pipe, filesystem access, or privileged system prompt;
   - `MOCK_CONTEXT_PATH`: an absolute path to a synthetic context based on
     `mock-database.example.json`;
   - `DATA_SOURCE_ID`: the same logical ID used in that context;
   - `DATABASE_PATH` and `RUNS_DIRECTORY`: trusted absolute host paths;
   - `TRUSTED_MODEL_NAME` and `TRUSTED_MODEL_SETTINGS_JSON`: the Pydantic AI model selection and
     safe settings used by DSA;
   - `DOCKER_IMAGE`: the immutable replay image in `name@sha256:<digest>` form; and
   - `REPORT_TO_MLFLOW`: optional observability through the already configured MLflow backend.
3. enable the Function and select **Private Data Chat** as the chat model.

The backend/provider credentials continue to come from its environment; no Valve accepts an API
key or provider endpoint. The clarifier receives neither the real database configuration nor the
original request's files, tools, system messages, or metadata.

The notebook is delivered as a persisted Open WebUI embed containing a fixed-name, base64-backed
download link. Before emitting it, the adapter reopens the exact retained file without following
symlinks and rechecks its byte length and SHA-256 digest. No host path is put in the chat.

`clarification-skill.md` is a host-owned system prompt loaded by the Pipe. It improves the
clarifier's behavior; the parsed contracts, confirmation digest, and trusted adapter enforce the
actual boundary.

See [THREAT_MODEL.md](THREAT_MODEL.md) for the boundary assumptions and demo limitations.
