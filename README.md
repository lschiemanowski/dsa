This repo contains two tools:

- `dsa`, an LLM-based data analysis tool for DuckDB databases
- `dsa chat`, a chat UI for using `dsa` while keeping the database and analysis on premises

Given a DuckDB database, `dsa` takes a request for a data analysis task along with a JSON Schema describing the answer. Depending on configuration, it can produce just the answer, or additionally a Jupyter notebook. The answer can also include up to 3 plots. In the initial request, `dsa` can also take guidance how to derive the answer. `dsa` is meant to be run by a trusted on premise model.

The analysis is performed by the model using tools to query the SQL database and to execute Python code. Python code is executed in a Docker sandbox without network access. Analysis uses a copy of the database, leaving the original unchanged.

`dsa chat` is a web UI for working with `dsa`. The user interacts with an untrusted, off premise model, which has no access to the database, but only a description of the database and some synthetic data rows. In this interaction, the untrusted model helps the user design a request to `dsa`. Once the user has reviewed and approved the request, the request is sent to `dsa`. The user receives the answer, along with any verified Jupyter notebook and plots. Requesting a notebook does not guarantee one: an answer can succeed without a verified derivation. Plots are available only with a verified derivation. This ends the conversation. The off premise model never sees any privileged data, unless supplied by the user.

Three demo databases and problem sets are provided on [Hugging Face](https://huggingface.co/datasets/lschiemanowski/dsa-datasets).

One of these datasets is a slice of the [SMARD](https://www.smard.de/en) electricity-market data for Germany and Luxembourg. It contains data on electricity generation by technology, consumption, forecasts, day-ahead prices and commercial cross-border net exports for the year 2024.

Here is a question a user may ask:

> How did Germany/Luxembourg’s electricity generation mix change month by month in 2024? Show a stacked bar chart of monthly generation in GWh, broken down by generation technology, and report each month’s renewable share as a percentage of total generation. Include a reproducible notebook with the calculations and plot.

Upon this, the off premise model asks the user to clarify their question, for example asking to decide how to group generation technology: by class (fossil, renewable, nuclear) or by individual technology (solar, coal, ...). Once the user has clarified their question, a request to `dsa` is drafted. This request is more concrete and specific than the user question, may contain guidance how to arrive at the result, and comes with a JSON Schema to return the answer in a structured manner. The user can review this request and approve it.

Here is a [video of the workflow](examples/smard/smard-monthly-generation-demo.mp4). The recording is played at twice the original speed, with most of the model waiting time removed.

You can look at the [complete answer](examples/smard/smard-monthly-generation-demo-answer.json) and [notebook](examples/smard/smard-monthly-generation-demo.ipynb) from the recorded session. The notebook contains the derivation of the result by `dsa`. Its purpose is for the user to verify that the derivation and therefore the result is correct.

To rerun the notebook, first download the database as described below. Open the notebook in Jupyter using a Python environment with DuckDB, pandas, Matplotlib and IPython installed. In its first code cell, replace `database_path = Path("database.duckdb")` with the absolute path to `data/smard-de-lu-2024/1.2.0/database/smard_de_lu_2024.duckdb` in your checkout, then run the cells in order. Plots are written to a `plots` directory relative to the notebook's working directory. Review the code before running it: execution in your own Jupyter environment is not protected by DSA's Docker sandbox.

The recorded session groups generation into renewables, nuclear and conventional generation. The standalone example below is a separate request that splits conventional generation into fossil and other generation; its output is therefore not expected to match the recording exactly.

<!-- For an inline GitHub video player, upload the MP4 in the web editor and insert the attachment URL here. -->

## How to run this

`dsa` can be run standalone or it can be invoked by the web UI. Run standalone, one LLM inference provider needs to be given, typically run locally. The web UI needs a second LLM provider, typically a hosted service. Here, I describe how to run `dsa` with `ollama` providing the local model and OpenRouter providing the hosted service.

You will need `uv`, Docker and Ollama installed, with Docker running and accessible to your user. Run the following commands from the root of a cloned checkout of this repo.

First, install the project with the chat dependencies:
```bash
uv sync --python 3.12 --extra chat --frozen
```
If Ollama is not already running as a service, start it in a separate terminal and leave it running:
```bash
ollama serve
```
In the original terminal, download the model and create its configuration:
```bash
ollama pull gemma4:e4b
ollama create dsa-gemma4 -f examples/smard/Modelfile
```
Then, in that same terminal, run
```bash
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=ollama
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
```
Download the demo datasets and problems:
```bash
uv run hf download lschiemanowski/dsa-datasets \
  --repo-type dataset \
  --revision 58958007cdf38eb9e563356f16ecd8011d5a3d67 \
  --local-dir data
```
This downloads all three demo datasets and their problem sets at the revision whose directory layout is expected by the example script.

Build the docker image for the sandbox:
```bash
docker build --tag dsa-python:local docker
export DSA_IMAGE=$(docker image inspect --format '{{.Id}}' dsa-python:local)
```
Finally, prepare the individual request and the chat config using the example script:
```bash
uv run python examples/smard/prepare.py --docker-image "$DSA_IMAGE"
```
Run an individual request:
```bash
uv run dsa run \
  --request examples/smard/local/task.json \
  --runs-directory examples/smard/local/runs \
  --docker-image "$DSA_IMAGE" \
  > examples/smard/local/result.json
```

To run the web UI, you first need to set an OpenRouter API key:
```bash
export OPENROUTER_API_KEY='your-openrouter-api-key'
```
Then run the UI:
```bash
uv run dsa chat --config examples/smard/local/chat.toml --host 127.0.0.1
```
This opens a web browser and shows the UI seen in the video. In this example configuration, DeepSeek V4 Flash through OpenRouter is used for clarification, while local Gemma 4 E4B performs the analysis. Edit `[clarifier]` and `[trusted]`, respectively, in `examples/smard/local/chat.toml` to change these models.

The recorded demo used DeepSeek V4 Flash for analysis of the public database as well. The local Gemma 4 E4B setup above has not yet been tested end to end.

## Benchmarks

We evaluated `dsa` with GPT-5.6 Luna, DeepSeek V4 Flash 0731, and GLM 5.3 Flash on the three demo datasets. For this comparison, we use 296 questions that ship with the demo datasets: 98 for Online Retail II, 100 for SMARD, and 98 for EEA air quality. Four questions whose rounding instructions were inaccurate have been dropped.

Each question was run once in four configurations, with or without guidance and with or without a requested notebook. The guidance is part of the problem sets.

| Guidance | Notebook requested | GPT-5.6 Luna | DeepSeek V4 Flash 0731 | GLM 5.3 Flash |
|---|---|---:|---:|---:|
| No | No | 256/296 (86.5%) | 257/296 (86.8%) | 221/296 (74.7%) |
| No | Yes | 250/296 (84.5%) | 235/296 (79.4%) | 180/296 (60.8%) |
| Yes | No | 257/296 (86.8%) | 255/296 (86.1%) | 225/296 (76.0%) |
| Yes | Yes | 251/296 (84.8%) | 233/296 (78.7%) | 195/296 (65.9%) |
| **All configurations** | | **1014/1,184 (85.6%)** | **980/1,184 (82.8%)** | **821/1,184 (69.3%)** |

The following are observed OpenRouter inference costs in USD for the same 296 questions per configuration. Amounts marked * have incomplete cost coverage and are partial sums, not complete totals. Luna's costs reflect its Flex routing. These figures exclude local compute and the `dsa-chat` clarification conversation.

| Guidance | Notebook requested | GPT-5.6 Luna | DeepSeek V4 Flash 0731 | GLM 5.3 Flash |
|---|---|---:|---:|---:|
| No | No | $0.3155 * | $0.7721 * | $0.3074 |
| No | Yes | $0.5329 * | $1.2674 * | $0.9333 * |
| Yes | No | $0.3055 * | $0.6313 | $0.3015 |
| Yes | Yes | $0.5104 * | $1.1805 | $0.7999 * |
| **All configurations** | | $1.6643 * | $3.8513 * | $2.3420 * |

Cost records cover all retained responses in 1,179/1,184 runs for Luna, 1,182/1,184 for DeepSeek, and 1,119/1,184 for GLM. Luna's five remaining runs have no retained model responses. DeepSeek has two retained responses without costs; GLM has 66. Missing costs are not treated as zero. The amounts are summed from provider-reported costs, including available costs for failed runs, rather than estimated from token prices.
