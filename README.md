# Kiln firing optimizer

An AI agent that plans a ceramic kiln firing: you describe the load in a sentence, and it finds the
lowest-energy schedule that meets your defect target. Two ML models score every candidate, and a safety
policy has to approve the result before it can be exported.

Runs on your own machine with no AWS account; the optional hosted LLM is Groq's API.
**AWS open-source projects used:** [Strands Agents SDK](https://strandsagents.com) (the agent loop and its four tools)
and [Cedar](https://www.cedarpolicy.com) (the safety rules, written as policy-as-code in `policies/safety.cedar`).

## How it works

```
Browser -> FastAPI -> Strands agent (LLM: Groq, Ollama or Bedrock)
                        |-- optimize : random search over safe schedules, scored by the energy + defect models
                        |-- validate : Cedar policy approves or rejects (fail-closed)
                        |-- predict, what_if
        <- firing curve, stage table, trade-off chart, approval badge
"Approve and export" is enabled only after Cedar approves, and the server re-checks before saving.
```

The agent never invents numbers: energy, defect rate and savings come from tool results.

## Quick start (no LLM needed)

```
uv sync
uv run uvicorn app.main:app
```

Open http://127.0.0.1:8000 and click an example. Without an LLM, the message box runs a **scripted demo**
(clearly labelled in the UI): the same `optimize` and `validate` tools, driven by a fixed script instead of an LLM.

## With a real LLM

Choose the language-model backend with `KILN_LLM`:

- `groq` (hosted): `uv add "strands-agents[openai]"`, then set `GROQ_API_KEY`. `GROQ_MODEL` defaults to `openai/gpt-oss-120b`.
- `ollama` (local, the default): install [Ollama](https://ollama.com), run `ollama pull llama3.1`, optionally set `OLLAMA_MODEL`.
- `bedrock`: set `BEDROCK_MODEL_ID` (needs an AWS account).

If the LLM is unavailable, the app falls back to the scripted demo and says why.

## Retrain the models (optional)

```
uv run python training/train_model.py
```

Trains an MLP and boosted trees for each target, picks the winner on the validation split, saves it to `models/`,
and prints the winner's test score.

| Target | Served model | Test MAE | Test R² |
|---|---|---|---|
| energy | _fill in from training output_ | | |
| defect rate | _fill in from training output_ | | |

## Tests

```
uv run python -m pytest -q
```

## Project layout

```
src/kiln/     config, models, safety (Cedar), optimizer, tools (Strands), agent, scripted demo, trace
policies/     safety.cedar
app/          FastAPI server + single-page UI
training/     train_model.py
simulator/    synthetic data generator
data/         earth_v2.csv
models/       trained model files
tests/        safety and scripted-mode tests
```

## Limitations (please read)

- The training data is **simulated**, so the numbers show the method, not a real kiln.
- The optimizer minimizes energy under a defect target; firing time is not constrained.
- The live agent needs an LLM backend (Groq key, local Ollama or Bedrock); otherwise it falls back to the scripted demo.
