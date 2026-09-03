# Private Data Chat

This directory contains an application built on top of DSA. It is deliberately outside
`src/dsa`: none of its UI, approval, identity, or deployment policy belongs to the reusable
analysis library.

The application lets a user refine a question with a capable but untrusted model using only a
small synthetic database description. After the user approves the exact quantitative question,
a trusted broker runs DSA against the real database with a separately configured trusted model.
The user receives the structured answer and, when DSA successfully validates a derivation, an
opaque download reference for the generated notebook.

## First milestone

The initial slice intentionally contains no web framework or Open WebUI integration. It defines
and tests the boundary that those adapters must use:

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

The in-memory store and fake executor are development aids. They are not production persistence
or a security boundary.

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

The single-process lock makes those statements true for `InMemoryProposalStore`. A production
store must implement the same transition checks atomically across every broker process.

## Planned adapters

The next vertical slice should add two thin adapters without weakening these contracts:

1. an external authenticated HTTP service that resolves `data_source_id` to the private database,
   trusted model, DSA policy, run directory, and artifact store; and
2. an Open WebUI Pipe/Action that sends only `MockDatabaseContext` to the untrusted model, displays
   the canonical proposal, obtains native user confirmation, and calls the service.

The Open WebUI component must not receive database paths, trusted-model credentials, Docker
access, the broker's data-source mapping, or unrestricted artifact access. A prompt skill may
improve clarification behavior, but the broker—not the prompt—enforces every security property.

See [THREAT_MODEL.md](THREAT_MODEL.md) for the boundary assumptions and failure policy.
