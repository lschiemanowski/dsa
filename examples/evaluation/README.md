# Current evaluation packs

Use `online-retail-ii.json`, `smard-de-lu-2024.json`, or
`eea-air-quality-six-cities-2018-2024.json` for the September revised problem sets.
Each contains 100 problems, with 25 replacements and 75 rewritten prompts relative
to its predecessor. The databases are unchanged.

These small locators pin Hugging Face commit
`f45a22879769fc02731f3dc55d126a8d6705d4b9` and individual manifest digests.
The data and cases live in `lschiemanowski/dsa-datasets`, under one folder per dataset:

```text
online-retail-ii/
  README.md
  PROBLEMS.md
  database-card.json
  database/
  pack.json
  cases.jsonl
```

SMARD and EEA follow the same layout, without a chat database card. The manifest
stays beside the database directory to preserve the loader's existing relative-path
contract. Internal manifest versions remain required metadata; public folder names
have no versions. Existing commit-pinned references still resolve historical data.

Download a current dataset with:

```bash
uv run --extra huggingface hf download lschiemanowski/dsa-datasets --repo-type dataset \
  --revision f45a22879769fc02731f3dc55d126a8d6705d4b9 \
  --include 'online-retail-ii/*' --local-dir data
```

All 300 case exports match the locally revised candidates byte for byte. Their
retained verification receipts record 244 replayed reference queries and 11
independent Python calculations; 56 predecessor answers were retained without
new independent recomputation. Difficulty labels remain provisional.

Reference answers and SQL are evaluator-only. Use `load_huggingface_evaluation_pack`
with a locator to verify content and build model inputs through the ordinary DSA
evaluation boundary.

## Running benchmarks

`dsa-benchmark plan`, `run`, and `report` use the pinned evaluation packs in a study.
Install the `huggingface` extra to load those packs. A v2 runtime JSON file contains
the workspace path and the Logfire opt-in. For example, save this as one line with a
trailing newline (paths can be relative to the runtime file):

```json
{"format":"dsa-benchmark-runtime/v2","report_to_logfire":false,"workspace_root":"./benchmark-workspace"}
```

To show experiments in Logfire, also install the `logfire` extra, set `LOGFIRE_TOKEN`,
and change `report_to_logfire` to `true` before running the study. Each completed cell
mirrors its scores after writing the local receipt. The experiment uses case and run
IDs, scores, and trace IDs, without exporting the questions, expected answers, actual
answers, or notebooks. Review the JSON run output for `experiment_reporting`; a telemetry
failure does not invalidate the receipt or change its scores.

For a new study replacing an MLflow study, remove the old runtime `datasets` bindings,
use a fresh workspace, and pin the new agent revision. Reports now use local cell
receipts (v2) and terminal records as evidence; they do not query a telemetry service.
The v4 report counts correct completed answers even if telemetry is unavailable, so
end-to-end rates can differ from v3. Keep prior reports and receipts as issued. Use
the original revision if you need to rebuild an older MLflow report.
