# Optional plots

Plots supplement the structured answer. They are available only through a verified
derivation and must be explicitly allowed in the task:

```json
"derivation": {"format": "dsa-derivation/v1", "allow_plots": true}
```

In Python, use `DerivationRequest(allow_plots=True)`. Omitting the flag leaves plots
disabled. Existing answer-only requests and derivations without plots remain supported.

In the chat, ask for a chart. The clarifier includes `allow_plots: true` in the proposal
shown for approval. That permission is included in the proposal digest and sent to the
trusted runner only after approval. A model may still return zero plots, or an answer
without a verified derivation if replay cannot succeed within the existing run budget.

## Model contract

The derivation may declare zero to three plots:

```json
"plots": [
  {"filename": "monthly-sales.png", "title": "Monthly sales", "caption": "Sales in GBP."}
]
```

Each filename must be unique, begin with an ASCII letter or digit, and contain only
letters, digits, underscores, or hyphens before `.png`. Titles are nonblank and at most
200 characters; optional captions are at most 1,000 characters.

Code cells receive `database_path` and `plot_directory`. Use Matplotlib with its Agg
backend, write declared PNGs beneath `plot_directory`, and assign the JSON answer to
`result` as before. Neither images nor filenames belong in the answer unless separately
required by the caller's answer schema. Matplotlib and Pillow are already in the DSA
Docker image; no runtime dependency installation or network access is needed.

## Verification and retention

- Replay starts from the pristine database inside the existing isolated executor.
- Only static PNGs are accepted: at most 2,048 pixels in either dimension, three files,
  and 5 MiB combined. Images are fully decoded after checking size and dimensions.
- Existing execution, storage, and artifact limits still apply; smaller configured
  limits can reject a plot even when it meets these maxima.
- Declared outputs use the existing transactional output snapshot/publication path.
  Missing or malformed files, oversized plots, and answer mismatches yield bounded
  retry feedback. Infrastructure failures stop replay as before.
- After answer validation, the exact replay-generated PNG bytes are embedded in the
  notebook. Its existing retained digest covers the images and their declarations.
  The notebook limit is 10 MiB to accommodate base64 image encoding and code.
- The chat reads the identity-verified notebook and offers its embedded images inline
  and as downloads with their declared filenames. Titles and captions are displayed
  as text, not active markup. The notebook remains the sole durable plot artifact;
  temporary replay files are removed.

Running the notebook again regenerates the images in a local `plots/` folder. Byte-for-byte
image equivalence across machines or library versions is not required. Verification
establishes executable reproduction of the answer and plot generation, **not** whether
the visualization is statistically appropriate or communicates the data fairly.

Real results, notebook content, and plots never return to the untrusted clarifier.
The chat still ends after the DSA result. Interactive charts, HTML/SVG output,
post-result editing, and automated chart-quality judging are out of scope.

MLflow reporting, when explicitly enabled, uploads the verified notebook as before;
that notebook now includes any plots, so they share the notebook's privacy boundary.
