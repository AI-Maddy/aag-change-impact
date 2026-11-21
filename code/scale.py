#!/usr/bin/env python3
"""Scalability of the deterministic core: time to compute I(C) (Eqs. 1-2) and claim status
(Eq. 3) on large synthetic graphs. Usage: python3 scale.py > scale.json"""
import json, random, time, sys
from collections import defaultdict

def build(n_ev, n_art, scope=12, p_dep=0.3, fanin=2, claims_per=4, rng=None):
    rng = rng or random.Random(0)
    S = [rng.sample(range(n_art), scope) for _ in range(n_ev)]
    D = [[rng.randrange(0, e) for _ in range(fanin)] if e > 10 and rng.random() < p_dep else [] for e in range(n_ev)]
    inv = defaultdict(list)
    for e, sc in enumerate(S):
        for a in sc: inv[a].append(e)
    cons = defaultdict(list)
    for e, ds in enumerate(D):
        for x in ds: cons[x].append(e)
    n_cl = n_ev // claims_per
    Ev = [rng.sample(range(n_ev), 3) for _ in range(n_cl)]
    sub = [[rng.randrange(c + 1, n_cl) for _ in range(2)] if c < n_cl - 1 and rng.random() < 0.5 else [] for c in range(n_cl)]
    return S, D, inv, cons, Ev, sub

def impact(C, inv, cons):
    I = set(); stack = []
    for a in C:
        for e in inv.get(a, ()):
            if e not in I: I.add(e); stack.append(e)
    while stack:
        x = stack.pop()
        for e in cons.get(x, ()):
            if e not in I: I.add(e); stack.append(e)
    return I

def support(Ev, sub, I):
    n = len(Ev); sup = [None] * n
    for c in range(n - 1, -1, -1):      # sub-claims have larger ids
        sup[c] = bool(Ev[c]) and all(e not in I for e in Ev[c]) and all(sup[s] for s in sub[c])
    return sup

out = []
for n_ev in (1_000, 10_000, 100_000, 1_000_000):
    rng = random.Random(1)
    n_art = n_ev * 3
    t0 = time.perf_counter(); S, D, inv, cons, Ev, sub = build(n_ev, n_art, rng=rng); tb = time.perf_counter() - t0
    for k in (1, 10, 100):
        ts, sizes = [], []
        for rep in range(5):
            C = rng.sample(range(n_art), k)
            t0 = time.perf_counter(); I = impact(C, inv, cons); sp = support(Ev, sub, I); ts.append(time.perf_counter() - t0)
            sizes.append(len(I))
        out.append({"evidence": n_ev, "artifacts": n_art, "claims": len(Ev), "changed": k,
                    "median_s": round(sorted(ts)[2], 4), "invalidated_median": sorted(sizes)[2], "build_s": round(tb, 2)})
        print(out[-1], file=sys.stderr)
json.dump(out, sys.stdout, indent=1)
