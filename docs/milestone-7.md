# Milestone 7: optional verified derivations

## Objective

Allow an individual task to ask for a concise, human-readable derivation alongside its
answer. The model may validate a proposed derivation before finalizing. A validated
derivation is accepted only after DSA replays it from the pristine source database; the
same verified content is retained as a deterministic Jupyter notebook for a human reviewer.
If validation does not succeed, the model may still submit its answer without a derivation.

Derivations are opt-in. Omitting the derivation request preserves the existing model
prompt, answer output shape, terminal-record schema version, evaluation input, and
artifact set. This makes the feature usable for selected standalone tasks or evaluation
cases without changing answer-only tasks.

## Request and model-output contract

`RunRequest.derivation` is either absent or the versioned request
`{"format":"dsa-derivation/v1"}`. When absent, the model returns only the caller's answer
as before. When present, DSA exposes `validate_derivation(derivation)`. A successful call
returns an opaque receipt bound to the replayed JSON result. The final output is:

```json
{
  "answer": {"value": 3},
  "derivation_receipt": "dvr_<sha256>"
}
```

`derivation_receipt` is optional (and nullable in the provider-facing strict schema). The
model should try the validation tool and may revise and retry within the ordinary run and
tool limits. It may omit the receipt when it cannot obtain one.

The derivation is a bounded sequence of Markdown and plain-Python cells. It begins with
an explanation, ends with code, and contains at most 24 cells and 64 KiB of source. Code
cells share one namespace. DSA supplies only `database_path`; the final namespace must
contain `result`. The format intentionally contains no notebook outputs, execution
counts, arbitrary metadata, raw exploratory transcript, or model reasoning trace.

The prompt asks for a verifier-oriented derivation: brief explanations and the direct
calculation needed to support the answer, with false starts and irrelevant exploration
omitted. This is a presentation artifact backed by executable evidence, not a claim to
capture the model's private chain of thought.

An evaluation-pack case opts in with the same optional `derivation` field. Its presence
is part of the canonical case and pack identity and is passed through the MLflow dataset
input. Packs may freely mix answer-only and derivation-enabled cases.

## Independent replay

Before model-visible exploration begins, DSA retains a private hard-linked identity of
the initial working database. This avoids a second initial database copy while ensuring
that later atomic database promotion cannot change the replay source. The caller's
database is never modified.

When the model calls `validate_derivation`, DSA runs the submitted code cells in order
through the already-injected isolated Python executor. Replay receives a disposable
working database linked from that pristine private source, not the database state left
by exploratory tools. The wrapper serializes `result` to a managed JSON output; DSA
loads it through the normal artifact boundary and binds its canonical JSON bytes to the
returned receipt. `final_answer` accepts that receipt only with the exact replayed answer.

The comparison is literal canonical JSON identity. In particular, `3` and `3.0` are not
the same result. A missing result, invalid JSON result, or model-authored execution failure
returns safe tool feedback and the model may try again. Such failures do not prevent an
answer-only final submission.
Executor unavailability, executor protocol failure, or private-workspace failure is an
infrastructure failure and is not presented to the model as repairable content.
Cancellation propagates after retaining the ordinary cancellation terminal record.

## Retained evidence and notebook

A derivation-enabled run uses terminal schema version 2, including when it succeeds
without a receipt, fails, or is cancelled. When its successful outcome includes a validated
derivation, it also contains a verification record binding these SHA-256 identities:

- canonical derivation content;
- canonical answer/result content;
- pristine source database;
- isolated runtime when the executor supplies one; and
- exact notebook bytes and length.

DSA projects the verified cells into `derivation.ipynb`. The notebook has deterministic
cell IDs and metadata, a trusted introductory cell, a trusted setup cell binding
`database_path = Path("database.duckdb")`, the exact model-authored cells, and a final
trusted `result` display cell. It is structurally validated with `nbformat` before
atomic no-overwrite publication. The notebook is not executed on the host.

The notebook is reported to MLflow as `dsa/derivation.ipynb` when reporting is enabled,
and `dsa-run` returns its path, digest, and byte length. Benchmark receipts carry the
derivation digest; immutable report construction verifies that identity against the
terminal record and hashes the exact local notebook before accepting the case. Report
v3's published result schema remains unchanged, so derivations add evidence without
changing existing accuracy metrics. Later LLM judges can consume the verified
derivation as a distinct evaluation input without weakening deterministic answer
scoring.

Answer-only successes continue to use terminal schema version 1 and create no notebook.
Answer-only, failed, and cancelled outcomes retain no notebook. A successfully replayed
candidate is retained only when its receipt accompanies the final answer.

## Acceptance

Deterministic tests cover opt-in and answer-only contracts, strict cell validation,
caller snapshot isolation, replay from pristine rather than exploratory state, exact
result identity, bounded retry and failure classification, deterministic valid notebook
generation, terminal invariants, evaluation-pack propagation, CLI projection, MLflow
artifact upload, report evidence correlation, tamper rejection, and cancellation.

The real Docker integration tier remains the authority for the sandbox implementation;
the derivation path deliberately reuses that executor rather than introducing a second
execution mechanism. Notebook execution by a human requires placing the exact source
database beside the notebook as `database.duckdb`; the retained source digest tells the
reviewer which bytes are required.
