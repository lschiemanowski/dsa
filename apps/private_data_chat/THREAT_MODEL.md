# Private Data Chat threat model

## Security objective

The clarification model may be malicious, compromised, prompt-injected, or simply wrong. It must
not gain access to the private database or influence privileged execution configuration. A user
must see and explicitly approve the exact quantitative request before any private analysis starts.

The trusted DSA model is allowed to analyze the selected private database only inside DSA's normal
tool and containment boundaries. Its structured answer and verified notebook are released to the
approving user.

## Trust boundaries

### Untrusted side

- chat content and bounded conversation history;
- the clarification model and everything it emits;
- the synthetic mock database context;
- the exact public database card pinned from the Hugging Face dataset, including model-only
  analysis notes;
- proposal text, answer schema, and optional analysis guidance until server validation; and
- future prompt skills or presentation code.

The untrusted side receives no real rows, private schema metadata beyond what an operator chose to
mock, filesystem path, provider secret, broker credential with broad scope, or DSA execution tool.
The configured clarifier must be a plain model connection without provider-side tools. The
application controls what is sent to the model, not what arbitrary software installed on the same
host can read.

### Trusted application side

- user or browser-session identity and conversation identity;
- the proposal store and approval state machine;
- the mapping from logical data-source IDs to private database paths;
- trusted-model selection, credentials, DSA policy, executor configuration, and run directories;
- canonical proposal hashing and expiry; and
- authorization of artifact downloads.

### Existing DSA boundary

DSA remains responsible for request validation, model/tool limits, database isolation, derivation
replay, terminalization, and retained notebook integrity. The application consumes its public API;
it does not reproduce those mechanisms.

When analysis guidance is enabled, it remains untrusted model output based only on synthetic data.
The user approves its exact text, and the adapter labels it as untrusted before sending it to the
trusted model. The application never executes suggested SQL or Python directly. The trusted model
may correct or ignore it, and only the trusted model's ordinary DSA derivation is replayed.

## Approval invariant

The approved digest covers the versioned proposal payload plus the user or browser-session
identity, conversation, and logical data source. The server stores that canonical content before presenting
it for approval. Execution reads the stored copy. It never accepts a fresh proposal body from the
confirmation event.

Proposal IDs are opaque, expire, and are scoped to the originating user and conversation. Approval
is idempotent. The HTTP workflow checks expiry and claims execution under the same broker lock.
Execution is at most once. A production multi-process deployment must enforce these transitions
with a database transaction or compare-and-swap, not an in-process lock.

## Configuration invariant

The host constructs the DSA request. The model-authored proposal cannot select or override:

- a database or filesystem path;
- trusted model or provider;
- credentials or endpoint;
- run limits or Docker configuration;
- run/artifact directories; or
- whether derivation validation is requested.

The host also decides whether analysis guidance is permitted. A disabled session rejects a model
proposal that contains guidance instead of silently removing model-authored content.

The first application profile requests a derivation. DSA may still return a successful answer
without a notebook when no validated derivation receipt was produced. The UI must label that state
honestly rather than fabricating or presenting an unverified notebook.

## Artifact invariant

The application keeps the retained notebook reference in its session. It reopens the notebook
without following symlinks and verifies the exact length and SHA-256 digest before delivering the
bytes. The HTTP API publishes a notebook URL only for the completed conversation and checks the
owning browser session on each download. The notebook is delivered under a fixed filename.
Artifact responses omit local paths.

## Failure policy

Provider and executor exception text is not returned across the application boundary. It is mapped
to stable codes. A failed analysis has no answer or notebook; it may reference a genuine retained
terminal record when the DSA adapter obtained one. The broker never invents terminal evidence.

Cancellation is recorded as `analysis_cancelled` before it is re-raised. If the process dies before
that transition, the record remains `running`; production recovery must reconcile it against DSA's
retained terminal evidence and must not automatically repeat an ambiguous paid execution.

## Deliberate demo limits

- user-account authentication, rate limits, and deployment hardening;
- durable multi-process transactions, crash recovery, and a durable artifact registry;
- protection from an operator putting sensitive values into the mock context; and
- semantic quality or privacy review of the final answer and notebook.

The HTTP API is intended for single-process, same-origin development on loopback. It accepts
loopback host names and checks the origin of browser writes. A server-issued session cookie
isolates conversations between browsers but does not authenticate a user account. The configured
data source is available to everyone allowed to access the app. A multi-user deployment must
restrict access to intended users.
