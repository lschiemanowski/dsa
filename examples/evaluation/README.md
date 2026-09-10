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
