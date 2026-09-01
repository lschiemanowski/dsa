# Milestone 6B: immutable benchmark study reports

## Objective

Add a read-only publication boundary that turns one completely verified Milestone 6A
study into one immutable standalone report. The report has a canonical machine-readable
representation and a deterministic Markdown projection. It does not run analyses,
retry cells, modify MLflow state, publish Git changes, or implement another evaluation
engine.

The canonical report is the evidence-bearing artifact. Markdown is a human-readable
projection of that same report, suitable for review or later inclusion in a README but
never inserted automatically.

## Publication unit and identity

One exact `dsa-benchmark-study/v1` contract produces at most one
`dsa-benchmark-report/v3` report for a given reporter revision. A report covers the
study's complete pack-model-repetition matrix; reports are not split by model, pack, or
repetition.

The report embeds the complete canonical study contract and retains both its study
digest and the exact clean Git revision of the reporting implementation. It also
contains the ordered evidence and derived results described below. Its compact,
recursively key-sorted JSON plus one trailing newline is canonical. The SHA-256 digest
of those exact bytes is the report identity and is returned by the publication command;
the digest is not embedded recursively in the JSON itself.

Neither representation contains a generation timestamp. Identical evidence processed
by the same reporting revision therefore produces identical JSON and Markdown bytes.
Publication time belongs to the later repository commit or release, not the benchmark
result identity.

The output basename is the safe study identity followed by the full study digest. The
publisher writes `<basename>.report-v3.json` and `<basename>.report-v3.md`. The
versioned names allow publication beside an earlier immutable report. Each file is
published atomically without overwriting different existing bytes. An interrupted
publication may be resumed only when every already-published file exactly matches the
bytes being produced; a conflict fails closed. Command success requires both complete
representations.

## Read-only evidence boundary

Publication accepts the same canonical study and runtime contracts used by Milestone
6A, plus an output directory. The runtime supplies the private benchmark workspace and
the exact Databricks dataset bindings but is not embedded wholesale because its local
workspace path is host-specific.

Before deriving any result, the publisher:

1. Revalidates and isolates the study and runtime contracts and derives the complete
   ordered cell plan from the canonical study.
2. Resolves and verifies every digest-pinned evaluation pack, including its manifest,
   cases, and database identity.
3. Requires exactly one canonical `cell.json` for every planned cell and rejects extra,
   missing, malformed, non-regular, conflicting, or mismatched receipts.
4. Verifies every receipt digest and its study, cell, pack, model, repetition, agent,
   image, dataset, evaluation-run, case-order, and run identities. For every prediction,
   it reads the exact bounded canonical terminal record from the final successful local
   attempt, hashes those bytes, and requires the digest, run identity, and projected
   outcome to agree with the receipt.
5. Reads each exact MLflow evaluation run by the ID retained in its receipt, requires
   an active `FINISHED` lifecycle state, and requires its benchmark tags to agree with
   the receipt and study.
6. Reads each successfully reported analysis run by its retained tracking-run ID and
   requires an active `FINISHED` lifecycle state and its safe DSA run identity to agree
   with the prediction. A prediction whose reporting projection itself failed remains
   a valid infrastructure observation and is not required to have a tracking run.
7. Recomputes scorer outcomes with the existing DSA scorer functions using the retained
   prediction plus the expected answer and scorer policy from the verified pinned pack,
   then cross-checks the corresponding finite aggregate metrics exposed by the MLflow
   evaluation run. A missing conditional metric is permitted only when that metric has
   no denominator.

Any unavailable required remote run, foreign record, identity contradiction, metric
contradiction, or incomplete study rejects publication. The publisher never discovers
"the latest" run by mutable tags, repairs remote state, substitutes another run, or
publishes a partial report. Hugging Face and Databricks access are read-only during this
operation.

The publisher does not reuse the execution preflight mechanically: publication does
not require the historical Docker image to remain installed and does not require the
current checkout revision to equal the study's agent revision. The study retains the
execution revision; the report separately retains the clean reporting revision.

## Canonical report content

`dsa-benchmark-report/v3` contains:

- the complete canonical study and its SHA-256 digest;
- the clean reporter Git revision;
- each pack's verified manifest digest, database ID and digest, case count, and exact
  repository and revision locator already bound by the study;
- each runtime dataset's safe name and the dataset ID and digest retained by every cell;
- one ordered cell record containing cell identity, receipt digest, evaluation-run ID,
  case count, and derived cell metrics;
- one ordered case outcome per cell containing only the case, run, terminal-record, and
  safe MLflow identities; accepted state; stable failure stage and code when present;
  reporting state; the six exact, policy, and failure scorer outcomes; and a bounded
  provider-cost summary derived from that case's verified terminal record;
- task-weighted aggregates for the complete study, for each model, for each pack-model
  pair, and for each cell; and
- bounded observed latency and usage summaries read from the exact reported analysis
  runs.

The report does not copy questions, schemas, model messages, answers, expectations, raw
failure messages, raw provider diagnostics, artifacts, credentials, provider endpoints,
Databricks hostnames, local paths, or mutable MLflow URLs. The embedded model settings
have already passed the study contract's secret-safe allowlist.

All collections have a canonical order. Cells retain the study order: pack identity,
model identity, then zero-based repetition. Case outcomes retain the verified pack case
order. Model and pack-model aggregates use the corresponding study order rather than
remote return order.

## Metrics and denominators

Every rate is represented by an integer numerator, an integer denominator, and a finite
rate when the denominator is positive. A zero denominator is represented explicitly and
has no rate; it is never coerced to zero. Counts are summed across underlying case
executions before rates are calculated. The publisher never averages cell percentages.

For each aggregation scope:

- **End-to-end exact success** counts predictions with an exact accepted answer and a
  successful reporting projection. Its denominator is every expected case execution.
- **Completion rate** counts schema-valid accepted answers, independent of exactness.
  Its denominator is every expected case execution.
- **Conditional exact accuracy** counts exact answers among schema-valid accepted
  answers. Its denominator is accepted answers only.
- **End-to-end policy success** counts accepted, successfully reported answers matching
  the pack's comparison policy. Its denominator is every expected case execution.
- **Conditional policy accuracy** counts policy matches among schema-valid accepted
  answers. Its denominator is accepted answers only.
- **Agent-failure rate** uses the existing `agent_failure` scorer. Its denominator is
  every expected case execution.
- **Infrastructure-failure rate** uses the existing `infrastructure_failure` scorer.
  Its denominator is every expected case execution.

Repetitions contribute separate case executions to these counts. Multiple packs are
task-weighted by their case counts rather than given equal pack weight. End-to-end policy
success is the pack-declared acceptance metric; end-to-end exact success remains the
strict cross-policy diagnostic. Completion, both conditional accuracies, agent failure,
and infrastructure failure are reported beside them.

The exact metrics always mean canonical JSON identity. A tolerance match can therefore
raise policy success without raising exact success. The case-level scorer outcomes and
aggregate counts must reproduce one another exactly.
An accepted answer may still coincide with an infrastructure failure when its reporting
projection failed; the metrics intentionally need not form a single exclusive
partition.

## Latency, usage, and cost

Latency and usage come only from explicit finite `dsa.elapsed_seconds` and
`dsa.usage.*` metrics on the exact per-analysis MLflow tracking runs named by the
receipts. Every summary states how many case executions supplied an observation and how
many did not. Totals and arithmetic means are calculated only over observed values;
missing observations are never treated as zero.

Provider cost is never estimated from a pricing table or scraped from raw trace
payloads. Report v3 derives it from the billed `cost` value that Pydantic AI retains in
each OpenRouter model response. Amounts are summed as exact decimal USD values from the
same canonical terminal bytes whose digest and outcome were already verified. Response
identities must be safe; reuse across terminal records rejects the report. Non-OpenRouter
responses, missing or invalid amounts, and duplicates within one case reduce explicit
coverage rather than becoming zero-cost generations. Parsed message histories are
discarded after this bounded evidence is extracted.

Each case, cell, model, pack-model pair, and complete study retains the observed subtotal,
source, currency, complete and incomplete case counts, and observed and unavailable
generation counts. `observed` means every response in every covered case supplied valid
evidence, `partial` means only an observed subtotal is known, and `unavailable` means no
amount was observed. MLflow receives a floating-point cost metric only for fully observed
future analysis runs; the exact terminal/report decimal remains authoritative. When that
metric exists, report publication cross-checks it against the terminal evidence. A
per-case mean is rendered only for fully observed cost; partial coverage exposes its
known subtotal but never treats unavailable cases as zero-cost cases.

## Markdown projection

The Markdown begins with the study identity, version, study digest, report JSON digest,
agent revision, reporter revision, pack identities, model identities, execution policy,
and repetition count. It then renders compact deterministic tables for:

- complete-study metrics;
- per-model metrics;
- per-pack-model metrics; and
- per-cell metrics and exact MLflow evaluation-run identities.

Each metric cell shows its integer numerator and denominator together with a stable
decimal rate when defined. Latency and token usage include observation coverage. Missing
conditional accuracy, usage, latency, or cost is written as `unavailable`, not `0`.

The Markdown makes no ranking, significance, confidence, difficulty, or generalization
claim. It links no mutable remote UI and contains no prose tailored to a particular
model's observed performance.

## CLI and acceptance

`dsa-benchmark report --study STUDY --runtime RUNTIME --output DIRECTORY` performs the
read-only validation and publication described above. It emits one bounded canonical
status object containing the study digest, report digest, and two output paths. Invalid
usage or configuration exits two, incomplete or contradictory evidence exits one,
success exits zero, and cancellation exits 130. It never starts a benchmark cell.
The output directory must be outside the private benchmark workspace so publication
cannot change the evidence inventory or break an idempotent rerun.

Deterministic tests cover strict report models, canonical ordering and identity,
mutation resistance, complete-matrix enforcement, receipt and pack verification,
terminal-byte hashing and run correlation, remote-ID correlation, completed active
MLflow lifecycle enforcement, MLflow mismatch rejection, all metric denominators, zero
accepted answers, task-weighted aggregation, repetition handling, observation coverage,
exact provider-cost aggregation and coverage, secret and answer exclusion,
deterministic Markdown, atomic no-overwrite publication,
idempotent recovery, bounded reads, CLI framing, and zero model execution.

Offline tests use fixed packs, receipts, and a narrow fake of the pinned MLflow client
surface. One opt-in Databricks acceptance test reads an exact deterministic test-model
study produced through Milestone 6A and verifies the native remote boundary without a
paid provider call. It must not create or modify datasets, runs, experiments, or traces.

Actual model selection and paid empirical studies, additional evaluation packs,
difficulty aggregation, confidence intervals, leaderboards, automatic README editing,
automatic Git or release publication, local MLflow, and user interfaces remain outside
Milestone 6B.
