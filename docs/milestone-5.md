# Milestone 5: versioned Hugging Face evaluation pack

## Objective

Publish and consume the existing twenty Online Retail II evaluation cases as one
versioned pack in the public Hugging Face dataset repository
`lschiemanowski/dsa-datasets`. The pack is a portable input to the existing native
Databricks MLflow evaluation path; it is not a second benchmark runner or result store.

The release lives beneath `online-retail-ii/1.0.0/` and contains one canonical case set,
one DuckDB database, a strict manifest, attribution, and database construction metadata.
The former core and generalization datasets survive only as migration provenance.

## Pack and locator

`pack.json` uses the strict format identifier `dsa-evaluation-pack/v1`. It records one
pack identity and semantic version, the license, one scorer policy, one database
identity/path/size/SHA-256, one canonical JSONL case path/count/size/SHA-256, and the two
predecessor MLflow dataset identities and export digests. A scorer is either exact JSON
or an explicit finite, bounded relative/absolute tolerance for floating-point leaves;
integer leaves always remain type-exact. The Online Retail II release uses exact JSON.

Each case contains only a stable case identity and version, the exact analytical
question, its Draft 2020-12 answer schema, the host-only expected answer, and descriptive
family and source-level metadata. It contains no host path, model configuration, run
policy, provider setting, credential, Docker setting, or MLflow destination.

Canonical case JSONL is UTF-8 with cases ordered by identity, one compact recursively
key-sorted finite JSON object per line, LF separators, and one final LF. Its file digest
is the canonical evaluation-export digest. Expected answers must satisfy their own
schemas. Case identities are unique and the manifest count must match exactly.

The DSA repository retains only a strict `dsa-huggingface-pack/v1` locator containing
the dataset repository identity, a full immutable Hugging Face commit SHA, the safe pack
subdirectory, and the expected manifest SHA-256. Release tags such as
`online-retail-ii-v1.0.0` are conveniences and are never the runtime authority.

## Loading and integrity

The optional Hugging Face loader delegates authenticated transport and version-aware
caching to `huggingface_hub`. It requests only the manifest, database, and case file at
the locator's exact repository, dataset type, revision, and safe paths. It bounds and
validates the manifest and case bytes, verifies every declared size and digest, rejects
unsafe or conflicting paths, and returns only a completely verified pack. Backend
exceptions are translated to stable safe error codes without retaining URLs, cache
paths, tokens, or raw remote diagnostics.

The returned database is an immutable cached source. Ordinary DSA execution continues
to make a private writable copy for each run. The loader never executes pack content,
imports Python from the dataset repository, or permits a custom remote dataset script.

## MLflow boundary

Native MLflow dataset inputs contain only case identity/version, database identity and
digest, question, and answer schema. The expected answer and scorer policy remain in the
native expectations field as separate canonical JSON strings, preserving JSON number
types across the managed Databricks dataset boundary. Descriptive pack metadata remains
in record tags. The host prediction adapter binds the verified local database path,
model configuration, run policy, and Python executor immediately before the ordinary
`run_analysis` call.

Consequently the same released evaluation dataset has stable record identity across
machines, models, policies, and cache locations. The model sees the question and answer
schema through the normal run request but never sees the expected answer or host-only
metadata.

## Publication and acceptance

Migration reads the prototype's frozen `online-retail-ii-core-v4` and
`online-retail-ii-generalization-v5` datasets once, verifies their established canonical
export digests, combines their exact twenty rows, and emits the new canonical pack. No
prototype importer or compatibility path remains after publication.

Publication creates one public Hugging Face commit containing the repository card and
complete versioned directory, tags that exact commit, then resolves it through a clean
cache and re-verifies every identity. The committed DSA locator is created only from that
verified immutable revision.

Deterministic tests cover strict models, canonical encoding, expectation validation,
duplicate identities, bounds, path safety, digest and size mismatches, exact downloader
arguments, sanitized failures, runtime-only evaluation bindings, and expectation
separation. One opt-in integration test downloads and verifies the exact public revision.

Model matrices, worker isolation, benchmark CLI orchestration, aggregate publication,
cost reporting, additional databases, local MLflow, paid model runs, and difficulty
normalization remain outside this milestone.

## Published release

The public release is
[`lschiemanowski/dsa-datasets`](https://huggingface.co/datasets/lschiemanowski/dsa-datasets)
at exact commit `897212ab5d9ad03631abccb5cc3e93f6a4396e65`, with convenience tag
`online-retail-ii-v1.0.0`. Its verified identities are:

- manifest: `703a821304f96a1ca7e301dcb5391a2c858ff4c265a2283d14739ef003e3e33d`
- twenty-case canonical export:
  `5cb9492096e7b8a7d3cbb5b033df42a51c30666eb1dcbb9aa1cf1faa17b06219`
- DuckDB database:
  `7439eff27b091d2cb4622ca9320e7f6aefccdc72af1895f983d838c7a518cbaf`

A clean-cache production-loader download reverified the manifest, ordered twenty case
identities, every expected answer, and all declared file sizes and digests before the
repository locator was added.
