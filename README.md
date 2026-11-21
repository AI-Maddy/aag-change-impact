# AAG change-impact benchmark and agents

Code and data for the Agentic Assurance Graph (AAG): a fail-closed core that decides which certification
evidence a change makes stale on a multicore hard real-time platform, a change-impact benchmark with ground
truth from re-running the analyses, baselines, and language-model agents that may only propose graph updates.

Python 3.10+. The core, benchmark, baselines and scoring use only the standard library.
`make_figs.py` needs matplotlib; the agent run needs `anthropic` (see `requirements.txt`).

## Layout

| Path | What it is |
|---|---|
| `code/aag.py`, `code/case_study.json` | Deterministic core on the brake-by-wire worked example (validity, closed invalidation, fail-closed support, scope checks, re-verification order, baselines) |
| `code/bench.py` | Benchmark: generated systems, real analyses (interference, RTA, MPU, HIL, TARA, unit/static, E2E, hazard review), 13 change types, trace-link / input-manifest / AAG methods, scope-error simulation |
| `code/scale.py` | Scalability of the core on synthetic graphs up to one million evidence items |
| `code/make_figs.py` | Figures and LaTeX table rows from the benchmark results |
| `code/a1_eval.py` | Report rendering (tool-native, vendor-aliased, prose) and the no-model A1 baselines |
| `code/agents.py` | Agents A1 (evidence ingestion) and A3 (re-verification planning), the model-alone baseline, task generation from exact benchmark scenarios, offline proposers and scoring |
| `code/score_agents.py`, `code/make_agent_table.py` | Re-score a model run from its raw answers; table rows |
| `notebooks/run_agents_colab.ipynb` | The notebook used for the model run (Message Batches API, pinned model id, budget cap); rebuilt by `notebooks/make_notebook.py` |
| `results/` | Benchmark results (3 seeds x 500 systems), scalability, worked example, A1 baselines, and the model run (`results/agents/`: raw answers, batch ids, scores) |

## Reproduce

```
cd code
python3 aag.py case_study.json                                        # worked example
for s in 1 2 3; do python3 bench.py 500 $s > ../results/bench_500_seed$s.json; done   # ~3 min
python3 scale.py > ../results/scale.json
python3 make_figs.py ../results /tmp/figs

# agents: re-score the released model run (no API key needed)
python3 score_agents.py ../results/agents/answers.jsonl ../results/agents/state.json > /tmp/scores.json
python3 make_agent_table.py /tmp/scores.json /tmp

# agents: dry run of the whole pipeline without a model
python3 agents.py tasks /tmp/r 2 1
python3 agents.py offline /tmp/r oracle
python3 agents.py score /tmp/r /tmp/r/answers_oracle.jsonl
```

A fresh model run uses `notebooks/run_agents_colab.ipynb` (Google Colab, an `ANTHROPIC_API_KEY` secret).
The released run used `claude-opus-5-5` at medium effort, 672 calls through the Message Batches API.
Model outputs are not deterministic, so a new run will differ in detail.

## Licence

MIT, see `LICENSE`.
