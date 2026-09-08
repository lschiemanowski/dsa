# Dataset-owned chat context

Private Data Chat's generic instructions remain in
`apps/private_data_chat/clarification-skill.md`. Dataset descriptions and explicitly
authored fake rows belong beside the database on Hugging Face:

```text
online-retail-ii/
  database-card.json
  synthetic-context.json
  database/
```

The checked-in `online-retail-ii.chat.toml.example` pins both files from
`lschiemanowski/dsa-datasets` at commit
`637fba846acb690d70324acd4ba52fae84c9a6a4`, with individual SHA-256 digests.
The synthetic export contains the two existing hand-authored fake examples—not real
database samples. No evaluation answers or reference SQL are included in chat context.

## Configuration and validation

Use the `[database_card]` and `[synthetic_context]` tables from the example. Both
references require an exact commit, safe relative path, and content digest; remote
context requires a card from the same repository and revision. Updating Hugging Face
`main` does not silently change a configured chat.

The loader reads at most 64 KiB per file, verifies the digest before parsing, requires
an explicit `synthetic: true` marker, and validates every represented relation and
its complete column set against the card. The dataset IDs must match. Synthetic
context may cover a subset of the card's relations; it may not invent extra columns.
Missing or invalid remote files stop preflight; they never trigger private database
sampling or a fallback to another context source. An explicit marker is a publication
contract, not proof that arbitrary data was actually fabricated: publishers must
review every value before publication.

For offline/custom deployments, `mock_context_path` still accepts a bounded local
synthetic JSON file. Configure exactly one of it or `[synthetic_context]`. The existing
local example remains an offline fixture, not the default retail deployment source.

## Database card v2

V2 adds `data_source_id` and replaces retail-specific coverage fields with up to 16
ordered facts:

```json
{
  "format": "dsa-database-card/v2",
  "data_source_id": "electricity",
  "coverage": {
    "facts": [
      {"label": "Period", "value": "Calendar year 2024"},
      {"label": "Regions", "value": "Germany and Luxembourg"}
    ]
  }
}
```

This excerpt omits the unchanged required title, summary, relations, primary relation,
and example questions. Fact values are bounded text or nonnegative integers. Labels
must be distinct. All displayed text is escaped against Markdown injection.
`analysis_notes` remain model-only: the user sees the card's overview, while the
clarifier receives the full card and synthetic context. The trusted runner separately
receives only the approved request, not this raw public context.

Legacy v1 cards and their historical pins remain readable. Retail's migration keeps
all approved descriptions, notes, relation definitions, and coverage values unchanged.
SMARD and EEA cards and synthetic examples are also published. Their cards describe
the recommended analysis relations with complete column lists and actual row counts.
SMARD has two fabricated quarter-hour rows; EEA has two fabricated sampling-point
rows and their matching city-hour summaries. No observation rows were sampled.

## SMARD and EEA pins

Both datasets' assets are pinned at Hugging Face commit
`c3dbcd3375a678eee85843f7ad738a1ded9edec3`. These are configuration fragments,
not complete chat configurations: also set the matching `data_source_id` and trusted
database path in your existing configuration. Keep your model settings unchanged.
Do not configure `mock_context_path` together with these remote references.

For SMARD, use `data_source_id = "smard-de-lu-2024"`:

```toml
[database_card]
repo_id = "lschiemanowski/dsa-datasets"
revision = "c3dbcd3375a678eee85843f7ad738a1ded9edec3"
path = "smard-de-lu-2024/database-card.json"
sha256 = "221aba0560d42e05cb9759ae88cfc51bad234db0fa34de28dc25ce93b19b9019"

[synthetic_context]
repo_id = "lschiemanowski/dsa-datasets"
revision = "c3dbcd3375a678eee85843f7ad738a1ded9edec3"
path = "smard-de-lu-2024/synthetic-context.json"
sha256 = "583efd06ebe60398006c660a23168bd58f93ad6193c62c005126cccaa23499b7"
```

For EEA, use `data_source_id = "eea-air-quality-six-cities-2018-2024"`:

```toml
[database_card]
repo_id = "lschiemanowski/dsa-datasets"
revision = "c3dbcd3375a678eee85843f7ad738a1ded9edec3"
path = "eea-air-quality-six-cities-2018-2024/database-card.json"
sha256 = "beabbb895a7794f797f7318772ea1518eba3ba82cde7de4fed144bc469f8990e"

[synthetic_context]
repo_id = "lschiemanowski/dsa-datasets"
revision = "c3dbcd3375a678eee85843f7ad738a1ded9edec3"
path = "eea-air-quality-six-cities-2018-2024/synthetic-context.json"
sha256 = "83d971bca7b6fa01aa70673ed6bc9b51a1fb9e68da8bf43afd0960b770bc8bbb"
```

Validate public downloads without making model calls:

```bash
DSA_HUGGINGFACE_TEST=1 uv run --all-groups --extra huggingface pytest \
  tests/private_data_chat/test_synthetic_context.py
```
