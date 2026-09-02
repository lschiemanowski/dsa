# Milestone 7: optional verified derivations

## Objective

Allow an individual task to request a concise, human-readable derivation alongside its
answer. A requested derivation is accepted only after DSA replays it from the pristine
source database and obtains exactly the same JSON value as the submitted answer. The
same verified content is retained as a deterministic Jupyter notebook for a human
reviewer.

Derivations are opt-in. Omitting the derivation request preserves the existing model
prompt, answer output shape, terminal-record schema version, evaluation input, and
artifact set. This makes the feature usable for selected standalone tasks or evaluation
cases without changing answer-only tasks.

## Request and model-output contract

`RunRequest.derivation` is either absent or the versioned request
`{"format":"dsa-derivation/v1"}`. When absent, the model returns only the caller's answer
as before. When present, DSA asks the model for this envelope:

```json
{
  "answer": {"value": 3},
  "derivation": {
    "format": "dsa-derivation/v1",
    "cells": [
      {
        "type": "markdown",
        "source": "Compute the requested value directly from the source table."
      },
      {
        "type": "code",
        "source": "import duckdb\ncon = duckdb.connect(str(database_path), read_only=True)\nresult = {\"value\": con.execute(\"select count(*) from events\").fetchone()[0]}"
      }
    ]
  }
}
```

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

After the answer satisfies its JSON Schema, DSA runs the submitted code cells in order
through the already-injected isolated Python executor. Replay receives a disposable
working database linked from that pristine private source, not the database state left
by exploratory tools. The wrapper serializes `result` to a managed JSON output; DSA
loads it through the normal artifact boundary and compares its canonical JSON bytes to
the submitted answer.

The comparison is literal canonical JSON identity. In particular, `3` and `3.0` are not
the same result. A mismatch, missing result, invalid JSON result, or model-authored
execution failure consumes one bounded derivation-validation attempt and returns safe
retry feedback. Exhaustion becomes a typed `derivation_validation` agent failure.
Executor unavailability, executor protocol failure, or private-workspace failure is an
infrastructure failure and is not presented to the model as repairable content.
Cancellation propagates after retaining the ordinary cancellation terminal record.

## Retained evidence and notebook

A derivation-enabled run uses terminal schema version 2, including when it fails or is
cancelled. Its successful outcome contains the validated derivation and a verification
record binding these SHA-256 identities:

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
Failed or cancelled runs retain no notebook, even if an earlier candidate derivation
was successfully replayed.

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
