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

- Open WebUI chat content and conversation history;
- the clarification model and everything it emits;
- the synthetic mock database context;
- proposal text and answer schema until server validation; and
- future prompt skills or presentation code.

The untrusted side receives no real rows, private schema metadata beyond what an operator chose to
mock, filesystem path, provider secret, broker credential with broad scope, or DSA execution tool.

### Trusted application side

- authenticated user and conversation identity;
- the proposal store and approval state machine;
- the mapping from logical data-source IDs to private database paths;
- trusted-model selection, credentials, DSA policy, executor configuration, and run directories;
- canonical proposal hashing and expiry; and
- authorization of artifact downloads.

### Existing DSA boundary

DSA remains responsible for request validation, model/tool limits, database isolation, derivation
replay, terminalization, and retained notebook integrity. The application consumes its public API;
it does not reproduce those mechanisms.

## Approval invariant

The approved digest covers the versioned proposal payload plus the authenticated user,
conversation, and logical data source. The server stores that canonical content before presenting
it for approval. Execution reads the stored copy. It never accepts a fresh proposal body from the
confirmation event.

Proposal IDs are opaque, expire, and are scoped to the originating user and conversation. Approval
is idempotent. Execution is at most once. A production multi-process deployment must enforce these
transitions with a database transaction or compare-and-swap, not an in-process lock.

## Configuration invariant

The host constructs the DSA request. The model-authored proposal cannot select or override:

- a database or filesystem path;
- trusted model or provider;
- credentials or endpoint;
- run limits or Docker configuration;
- run/artifact directories; or
- whether derivation validation is requested.

The first application profile requests a derivation. DSA may still return a successful answer
without a notebook when no validated derivation receipt was produced. The UI must label that state
honestly rather than fabricating or presenting an unverified notebook.

## Artifact invariant

External responses contain an opaque artifact ID, SHA-256 digest, byte length, and media type—not
a local path. A later artifact endpoint must reauthorize the current user against the proposal and
stream the exact retained bytes with a fixed content disposition. Artifact IDs must not grant
ambient access by possession alone.

## Failure policy

Provider and executor exception text is not returned across the application boundary. It is mapped
to stable codes. A failed analysis has no answer or notebook; it may reference a genuine retained
terminal record when the DSA adapter obtained one. The broker never invents terminal evidence.

Cancellation is recorded as `analysis_cancelled` before it is re-raised. If the process dies before
that transition, the record remains `running`; production recovery must reconcile it against DSA's
retained terminal evidence and must not automatically repeat an ambiguous paid execution.

## Out of scope for this milestone

- HTTP authentication, CSRF protection, rate limits, and deployment hardening;
- durable multi-process transactions and crash recovery;
- real DSA/database and artifact-serving adapters;
- Open WebUI Pipe, Action, and file-download integration;
- protection from an operator putting sensitive values into the mock context; and
- semantic quality or privacy review of the final answer and notebook.

The eventual HTTP adapter must also authorize the requested `data_source_id` for the current user;
syntactic validation of a logical ID is not data-source authorization.
