#!/usr/bin/env python3
"""Build run_agents_colab.ipynb: a self-contained Colab notebook that embeds the evaluation code
(bench.py, a1_eval.py, agents.py) so it needs no repository access. Re-run after editing the code:
    python3 make_notebook.py
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}

cells = [md("""
# AAG agents on a real model

Runs the AAG language-model agents (A1 evidence ingestion, A3 re-verification planning, and a
model-alone change-impact baseline) on a sample of the 19,500 benchmark scenarios, then scores them.

**Before you start:** the Colab secret `ANTHROPIC_API_KEY` must exist with notebook access on (key icon on the left).

**To run:** Runtime > Run all. Allow Google Drive access when asked. A CPU runtime is enough.
It sends requests through the Message Batches API (half price) and stops at the budget below.
Results go to `MyDrive/aag-agent-eval/`. If Colab disconnects, run all again: it resumes where it stopped
and does not pay twice for finished batches.
"""),
code("""
BUDGET_USD = 18.0          # hard ceiling for this notebook (pilot + main run)
MODEL = "claude-opus-5-5"  # pinned model id
!pip -q install anthropic==2.11.0
"""),
code("""
import os, sys, json, time
from google.colab import drive, userdata
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/aag-agent-eval'
os.makedirs(OUT, exist_ok=True)
os.makedirs('/content/aag', exist_ok=True); os.chdir('/content/aag'); sys.path.insert(0, '/content/aag')
os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
print('Drive folder:', OUT)
"""),
]
for fn in ("bench.py", "a1_eval.py", "agents.py"):
    src = open(os.path.join(CODE, fn)).read()
    cells.append(code(f"%%writefile /content/aag/{fn}\n" + src))

cells += [code("""
import anthropic, agents
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
agents.MODEL = MODEL
client = anthropic.Anthropic()
m = client.models.retrieve(MODEL)          # free; fails early if the key or model id is wrong
print('Model OK:', m.id, '| anthropic', anthropic.__version__)

K1MAX, K3MAX = 10, 4                       # largest sample the planner may choose (systems per seed)
A1, A3 = agents.gen_tasks(K1MAX, K3MAX)
TASKS = {t['id']: t for t in A1 + A3 if t.get('prompt')}
idx = lambda t: int(t['sid'].split('sys')[1])
print(len(TASKS), 'model tasks available')

STATE = f'{OUT}/state.json'; ANS = f'{OUT}/answers.jsonl'
state = json.load(open(STATE)) if os.path.exists(STATE) else {'batches': {}}
def save_state(): json.dump(state, open(STATE, 'w'), indent=1)
def answers():
    out = {}
    if os.path.exists(ANS):
        for l in open(ANS):
            r = json.loads(l); out[r['id']] = r
    return out
def spent(): return sum(r['cost_usd'] for r in answers().values())
"""),
code("""
def run_batch(name, ids):
    \"\"\"Submit (or resume) one batch, wait for it, append results to answers.jsonl.\"\"\"
    done = answers()
    ids = [i for i in ids if i not in done or done[i]['status'] == 'errored']
    if not ids:
        print(name, 'already done'); return
    if name not in state['batches']:
        reqs = [Request(custom_id=f'r{n}', params=MessageCreateParamsNonStreaming(**agents.request_params(TASKS[i], MODEL)))
                for n, i in enumerate(ids)]
        b = client.messages.batches.create(requests=reqs)
        state['batches'][name] = {'id': b.id, 'map': {f'r{n}': i for n, i in enumerate(ids)}}
        save_state(); print(name, 'submitted', b.id, len(ids), 'requests')
    info = state['batches'][name]
    while True:
        b = client.messages.batches.retrieve(info['id'])
        if b.processing_status == 'ended': break
        print(time.strftime('%H:%M:%S'), name, b.request_counts); time.sleep(60)
    with open(ANS, 'a') as f:
        for r in client.messages.batches.results(info['id']):
            tid = info['map'][r.custom_id]
            row = {'id': tid, 'status': r.result.type, 'answer': None, 'text': None, 'stop_reason': None,
                   'usage': {}, 'cost_usd': 0.0, 'model': None}
            if r.result.type == 'succeeded':
                msg = r.result.message
                text = ''.join(c.text for c in msg.content if c.type == 'text')
                u = msg.usage.model_dump(exclude_none=True)
                u = {k: v for k, v in u.items() if isinstance(v, (int, float))}
                row.update(text=text, stop_reason=msg.stop_reason, usage=u, model=msg.model,
                           cost_usd=agents.cost_usd(u, batch=True),
                           answer=agents.parse_answer(text) if msg.stop_reason == 'end_turn' else None)
            f.write(json.dumps(row) + '\\n')
    print(name, 'finished; total spent so far $%.2f' % spent())
"""),
md("## 1. Pilot (one system) to measure the real cost per call"),
code("""
pilot = [i for i, t in TASKS.items() if t['sid'] == 'seed1.sys0']
run_batch('pilot', pilot)
A = answers()
per = {}
for ag in ('a1', 'impact', 'a3'):
    c = [A[i]['cost_usd'] for i in pilot if TASKS[i]['agent'] == ag and i in A]
    per[ag] = sum(c) / max(1, len(c))
print('mean cost per call (USD):', {k: round(v, 4) for k, v in per.items()})
fails = [i for i in pilot if A.get(i, {}).get('answer') is None]
print('pilot calls without a usable answer:', len(fails), fails[:5])
"""),
md("## 2. Choose the sample size that fits the budget, then run it"),
code("""
def plan_cost(k1, k3):
    ids = [i for i, t in TASKS.items() if (t['agent'] == 'a1' and idx(t) < k1) or (t['agent'] != 'a1' and idx(t) < k3)]
    todo = [i for i in ids if i not in answers()]
    return ids, sum(per[TASKS[i]['agent']] for i in todo) * 1.25   # 25% safety margin
plans = [(4, 1), (6, 1), (6, 2), (8, 2), (8, 3), (10, 3), (10, 4)]
left = BUDGET_USD - spent()
chosen = None
for k1, k3 in plans:
    ids, c = plan_cost(k1, k3)
    print(f'K1={k1} K3={k3}: {len(ids)} calls, est. ${c:.2f} more (left ${left:.2f})')
    if c <= left: chosen = (k1, k3, ids)
assert chosen, 'Budget too small even for the smallest plan; raise BUDGET_USD.'
if state.get('plan'):                      # a resumed run keeps the plan it started with
    K1, K3 = state['plan']; ids = plan_cost(K1, K3)[0]
else:
    K1, K3, ids = chosen
state['plan'] = [K1, K3]; save_state()
print('Running K1 =', K1, 'and K3 =', K3, 'systems per seed (x3 seeds)')
run_batch(f'main_{K1}_{K3}', ids)
run_batch(f'retry_{K1}_{K3}', [i for i in ids if answers().get(i, {}).get('status') == 'errored'])
print('TOTAL SPENT: $%.2f' % spent())
"""),
md("## 3. Score and save"),
code("""
A = answers()
ans = {i: r['answer'] for i, r in A.items()}
a1 = [t for t in A1 if idx(t) < K1]; a3 = [t for t in A3 if idx(t) < K3]
scen = agents.sample_scenarios(K1)
S = agents.score(a1, a3, ans, scen)
S['meta'] = {'model': MODEL, 'served_models': sorted({r['model'] for r in A.values() if r['model']}),
             'effort': agents.EFFORT, 'K1': K1, 'K3': K3, 'calls': len(A), 'spent_usd': round(spent(), 2),
             'stop_reasons': {s: sum(1 for r in A.values() if r['stop_reason'] == s) for s in {r['stop_reason'] for r in A.values()}},
             'anthropic_sdk': anthropic.__version__, 'date': time.strftime('%Y-%m-%d')}
json.dump(S, open(f'{OUT}/scores.json', 'w'), indent=1, default=float)
a, e, i3, p3 = S['a1']['ALL'], S['a1_end_to_end'], S['impact']['ALL'], S['a3']['ALL']
print(f\"A1: {int(a['n'])} proposals, omission rate {a['omission_rate']:.3f}, exact {a['exact_rate']:.2f}, \"
      f\"rejected by checks {a['rejection_rate']:.2f}, bad proposals rejected {a['bad_rejected_share']}\")
print(f\"A1 end-to-end recall: true scopes {e['true.recall']:.3f}, agent no checks {e['agent_nochecks.recall']:.3f}, agent + checks {e['agent_checks.recall']:.3f}\")
print(f\"Model alone (impact): recall {i3['recall']:.3f}, precision {i3['precision']:.3f}, AAG core recall {i3['core_recall']:.3f}\")
print(f\"A3 plans valid {p3['plan_valid_rate']:.2f}, causal paths correct {p3['path_ok_rate']:.2f}, extras that caught core misses {int(p3['extra_stale'])}\")
print('Saved to', OUT, '- tell Claude it has finished.')
"""),
]
nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(os.path.join(HERE, "run_agents_colab.ipynb"), "w"), indent=1)
print("wrote run_agents_colab.ipynb")
