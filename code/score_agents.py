#!/usr/bin/env python3
"""Re-score the Colab agent run from its raw answers.
Usage: python3 score_agents.py ../results/agents/answers.jsonl ../results/agents/state.json > ../results/agents/scores.json"""
import json, sys
import agents

ans_rows = [json.loads(l) for l in open(sys.argv[1])]
state = json.load(open(sys.argv[2]))
K1, K3 = state["plan"]
A1, A3 = agents.gen_tasks(K1, K3)
idx = lambda t: int(t["sid"].split("sys")[1])
a1 = [t for t in A1 if idx(t) < K1]; a3 = [t for t in A3 if idx(t) < K3]
rows = {}
for r in ans_rows:
    rows[r["id"]] = r            # later rows (retries) win
ids = {t["id"] for t in a1 + a3 if t.get("prompt")}
assert ids <= set(rows), f"missing answers: {len(ids - set(rows))}"
S = agents.score(a1, a3, {i: r["answer"] for i, r in rows.items()}, agents.sample_scenarios(K1))
S["meta"] = {"model": agents.MODEL, "served_models": sorted({r["model"] for r in rows.values() if r["model"]}),
             "effort": agents.EFFORT, "K1": K1, "K3": K3, "calls": len(rows),
             "spent_usd": round(sum(r["cost_usd"] for r in rows.values()), 2),
             "stop_reasons": {s: sum(1 for r in rows.values() if r["stop_reason"] == s)
                              for s in {r["stop_reason"] for r in rows.values()}},
             "input_tokens": sum(r["usage"].get("input_tokens", 0) for r in rows.values()),
             "output_tokens": sum(r["usage"].get("output_tokens", 0) for r in rows.values()),
             "date": "2026-11-20", "batches": [b["id"] for b in state["batches"].values()]}
tp = fp = 0
for t in a3:
    if t["agent"] == "impact":
        inv, st = set(t["inv"]), set(t["stale"]); tp += len(inv & st); fp += len(inv - st)
S["core_precision_impact"] = tp / (tp + fp)
json.dump(S, sys.stdout, indent=1, default=float)
