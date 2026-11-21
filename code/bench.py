#!/usr/bin/env python3
"""Change-impact benchmark for the Agentic Assurance Graph (AAG).

Ground truth is semantic: after a change, every analysis is re-run on the changed system
and an evidence item counts as stale iff its output differs. Methods only see artifact
hashes (syntactic). Analyses are real but small:
  - multicore interference: per-access delay bound from round-robin arbitration on the
    shared island SRAM port (slot latency x number of contending cores);
  - response-time analysis per core: fixed-priority (rate-monotonic) RTA with
    interference-inclusive execution times C' = C + m*d (m = accesses per job);
  - MPU partition check over island partitions (overlap = violation);
  - HIL latency test of the brake chain (RTA with average-case interference, 10 us resolution);
  - TARA: risk per threat from attack feasibility, controls and CAN access surface;
  - unit tests (behaviour version) and static analysis (code version) per task;
  - end-to-end protection tests (E2E library and protected CAN messages).
No language model is involved. Usage: python3 bench.py [n_systems] [seed] > results.json
"""
import hashlib, json, math, random, sys, copy
from collections import defaultdict

ISLAND = ["I0", "I1"]
APP = ["A0", "A1", "A2", "A3"]
CORES = ISLAND + APP
# Kramer et al. (WATERS 2015) period shares for periodic runnables, angle-synchronous excluded
PERIODS_MS = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
PERIOD_W = [3, 2, 2, 25, 25, 3, 20, 1, 4]
COST_H = {"interference.analytic": 2, "interference.measured": 8, "rta": 0.5, "mpu": 0.1,
          "hil": 16, "unit": 0.1, "static": 0.2, "e2e": 0.5, "tara": 24, "review": 8}

def hsh(v):
    return hashlib.sha256(repr(v).encode()).hexdigest()[:16]

def uunifast(n, u, rng):
    out, s = [], u
    for i in range(1, n):
        nxt = s * rng.random() ** (1.0 / (n - i))
        out.append(s - nxt); s = nxt
    out.append(s)
    return out

# ----------------------------------------------------------------------------- system
def gen_system(rng):
    tasks = {}
    for core in CORES:
        n = rng.randint(3, 6)
        u = rng.uniform(0.35, 0.55) if core in ISLAND else rng.uniform(0.4, 0.65)
        for i, ui in enumerate(uunifast(n, u, rng)):
            T = rng.choices(PERIODS_MS, PERIOD_W)[0] * 1000
            name = f"{core}t{i}"
            tasks[name] = dict(core=core, C=max(5.0, round(ui * T, 1)), T=T,
                               r=round(rng.uniform(0.3, 2.0), 2),
                               sram=(core in ISLAND) or rng.random() < 0.25,
                               crit=("D" if core == "I0" else "B") if core in ISLAND else "QM",
                               code=0, behav=0, can=False)
    # the brake controller: highest-rate task on I0
    tasks["BrakeCtl"] = dict(core="I0", C=round(rng.uniform(150, 400), 1), T=5000, r=1.0, sram=True,
                             crit="D", code=0, behav=0, can=True)
    tasks["WSpeed"] = dict(core="I1", C=round(rng.uniform(100, 300), 1), T=5000, r=0.8, sram=True,
                           crit="B", code=0, behav=0, can=True)
    parts, base = {}, 0
    for t in sorted(x for x in tasks if tasks[x]["core"] in ISLAND):
        size = rng.choice([4, 8, 16]) * 1024
        parts[t] = (base, size); base += size + rng.choice([0, 1024, 4096])
    can = {f"msg{i}": rng.random() < 0.6 for i in range(rng.randint(4, 8))}
    can["BrakeReq"] = True
    return dict(tasks=tasks, parts=parts, ic=dict(L=round(rng.uniform(0.02, 0.06), 3),
                app_port=rng.random() < 0.5), controls=["SecOC"], can=can, e2e_code=0, item=0,
                dma=False)

# ----------------------------------------------------------------------------- analyses
def interference(s):
    """Per-access delay bound on each core: round-robin arbitration, each other requester j
    delays an access by at most one slot L, weighted by its access intensity min(1, rho_j)
    (rho_j = SRAM accesses per microsecond issued by core j)."""
    rho = defaultdict(float)
    for t, d in s["tasks"].items():
        if d["sram"] and (d["core"] in ISLAND or s["ic"]["app_port"]):
            rho[d["core"]] += d["C"] / d["T"] * d["r"]
    if s.get("dma"):
        rho["DMA"] = 0.8   # an app-side DMA engine reaching the SRAM port (in no template)
    return {k: round(s["ic"]["L"] * sum(min(1.0, v) for j, v in rho.items() if j != k), 4)
            if k in rho else 0.0 for k in CORES}

def rta(s, core, d, scale=1.0):
    ts = sorted([t for t, x in s["tasks"].items() if x["core"] == core],
                key=lambda t: (s["tasks"][t]["T"], t))
    out, hp = [], []
    for t in ts:
        x = s["tasks"][t]
        c = x["C"] + x["r"] * x["C"] * d * scale
        R = c
        for _ in range(1000):
            Rn = c + sum(math.ceil(R / s["tasks"][j]["T"]) * cj for j, cj in hp)
            if Rn > x["T"]:
                R = None; break
            if abs(Rn - R) < 1e-9:
                break
            R = Rn
        out.append((t, "miss" if R is None else round(R, 1)))
        hp.append((t, c))
    return tuple(out)

def mpu(s):
    p = sorted(s["parts"].items())
    viol = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            (a, (ba, sa)), (b, (bb, sb)) = p[i], p[j]
            if ba < bb + sb and bb < ba + sa:
                viol.append((a, b))
    return (tuple(t for t, _ in p), tuple(viol))

def hil(s, d):
    r = dict(rta(s, "I0", d["I0"], scale=0.5)).get("BrakeCtl")
    return "miss" if r == "miss" else int(round((r + 300) / 10.0)) * 10

def tara(s):
    risks = []
    surface = sum(1 for x in s["tasks"].values() if x["core"] in ISLAND and x["can"])
    for m, prot in sorted(s["can"].items()):
        feas = 3 - ("IDS" in s["controls"]) - (1 if prot and "SecOC" in s["controls"] else 0)
        feas += 1 if surface > 2 else 0
        risks.append((m, max(feas, 0) * (3 if m == "BrakeReq" else 2)))
    risks.append(("OTA", 2 if "SecureBoot" in s["controls"] else 4))
    return tuple(risks)

def outputs(s, ev_ids):
    d = interference(s)
    out = {}
    for e in ev_ids:
        kind, _, arg = e.partition(":")
        if kind == "INT":
            out[e] = tuple(sorted(d.items()))
        elif kind == "RTA":
            out[e] = rta(s, arg, d[arg])
        elif kind == "MPU":
            out[e] = mpu(s)
        elif kind == "HIL":
            out[e] = hil(s, d)
        elif kind == "UT":
            out[e] = s["tasks"][arg]["behav"] if arg in s["tasks"] else None
        elif kind == "SA":
            out[e] = s["tasks"][arg]["code"] if arg in s["tasks"] else None
        elif kind == "E2E":
            out[e] = (s["e2e_code"], tuple(sorted(m for m, p in s["can"].items() if p)))
        elif kind == "TARA":
            out[e] = tara(s)
        elif kind == "HARA":
            out[e] = s["item"]
    return out

def critical(e, old, new):
    """A change in output that worsens a verdict."""
    kind = e.split(":")[0]
    if kind == "RTA":
        return sum(1 for _, r in new if r == "miss") > sum(1 for _, r in old if r == "miss")
    if kind == "MPU":
        return len(new[1]) > len(old[1])
    if kind == "HIL":
        return new == "miss" or (old != "miss" and new > old)
    if kind == "TARA":
        o, n = dict(old), dict(new)
        return any(k not in o or n[k] > o[k] for k in n)
    if kind in ("UT", "E2E"):
        return True
    return False

# ----------------------------------------------------------------------------- artifacts
def artifacts(s, file_level=False):
    a = {}
    T = s["tasks"]
    rows = {k: tuple(sorted(t for t in T if T[t]["core"] == k)) for k in CORES}
    tim = {t: (x["C"], x["T"], x["crit"], x["can"]) for t, x in T.items()}
    mem = {t: (x["r"], x["sram"]) for t, x in T.items()}
    tp = {t: (tim[t], mem[t]) for t in T}
    if file_level:
        a["alloc"] = tuple(sorted(rows.items())); a["tp"] = tuple(sorted(tp.items()))
        a["part"] = tuple(sorted(s["parts"].items()))
    else:
        # entry-level keys; every table also gets an index key so that adding or removing
        # an entry is visible to scopes that cover the whole table
        for k in CORES: a[f"alloc.{k}"] = rows[k]
        for t in T: a[f"tim.{t}"] = tim[t]; a[f"mem.{t}"] = mem[t]
        for t, p in s["parts"].items(): a[f"part.{t}"] = p
        a["part.index"] = tuple(sorted(s["parts"]))
    for t, x in T.items(): a[f"code.{t}"] = x["code"]
    a["ic"] = (s["ic"]["L"], s["ic"]["app_port"])
    a["arch.sec"] = (tuple(sorted(s["controls"])), tuple(sorted(t for t in T if T[t]["can"])))
    a["can"] = tuple(sorted(s["can"].items()))
    a["code.e2e"] = s["e2e_code"]
    a["rig"] = 0; a["item"] = s["item"]
    isl = [t for t in T if T[t]["core"] in ISLAND]
    a["img.island"] = hsh((rows["I0"], rows["I1"], [(tp[t], T[t]["code"]) for t in sorted(isl)],
                           sorted(s["parts"].items()), a["ic"]))
    a["dma"] = s.get("dma", False)
    a["img.app"] = hsh(([rows[k] for k in APP], [(tp[t], T[t]["code"]) for t in sorted(T) if t not in isl], a["dma"]))
    return {k: hsh(v) for k, v in a.items()}

def keys(s, file_level, kind, arg=None):
    """Artifact keys of a kind; file_level maps entry keys to their file."""
    T = s["tasks"]
    if kind == "alloc":
        return ["alloc"] if file_level else [f"alloc.{k}" for k in ([arg] if arg else CORES)]
    if kind in ("tim", "mem", "tp"):
        ts = [t for t in T if (arg is None or T[t]["core"] == arg)]
        if file_level:
            return ["tp"]
        sub = ["tim", "mem"] if kind == "tp" else [kind]
        return [f"{x}.{t}" for t in ts for x in sub]
    if kind == "part":
        return ["part"] if file_level else [f"part.{t}" for t in s["parts"]] + ["part.index"]
    raise KeyError(kind)

def evidence(s, int_variant, file_level=False):
    """Evidence items with manifest M (what the producing tool reads), template-required and
    declared extra scope X, and consumed evidence D."""
    T = s["tasks"]; ev = {}
    K = lambda kind, arg=None: keys(s, file_level, kind, arg)
    ports = CORES if s["ic"]["app_port"] else ISLAND   # cores wired to the SRAM port
    interf_tmpl = [x for k in ports for x in K("alloc", k) + K("tp", k)] + ["ic"]
    if int_variant == "analytic":
        ev["INT"] = dict(type="interference.analytic", M=interf_tmpl + ["dma"], X=[], D=[])
    else:
        ev["INT"] = dict(type="interference.measured", M=["img.island", "img.app", "rig"], X=interf_tmpl, D=[])
    for k in CORES:
        ev[f"RTA:{k}"] = dict(type="rta", M=K("alloc", k) + K("tp", k) + ["report.INT"], X=[], D=["INT"])
    ev["MPU"] = dict(type="mpu", M=K("part"), X=[], D=[])
    ev["HIL"] = dict(type="hil", M=["img.island", "rig"], X=interf_tmpl, D=[])
    for t in T:
        ev[f"UT:{t}"] = dict(type="unit", M=[f"code.{t}"], X=[], D=[])
        ev[f"SA:{t}"] = dict(type="static", M=[f"code.{t}"], X=[], D=[])
    ev["E2E"] = dict(type="e2e", M=["code.e2e", "can"], X=[], D=[])
    ev["TARA"] = dict(type="tara", M=[], X=["arch.sec", "can"], D=[])
    ev["HARA"] = dict(type="review", M=[], X=["item"], D=[])
    for e in ev:
        ev[e]["M"] = sorted(set(ev[e]["M"])); ev[e]["X"] = sorted(set(ev[e]["X"]) - set(ev[e]["M"]))
    return ev

def template(s, etype, file_level=False):
    K = lambda kind, arg=None: keys(s, file_level, kind, arg)
    ports = CORES if s["ic"]["app_port"] else ISLAND
    it = set(x for k in ports for x in K("alloc", k) + K("tp", k)) | {"ic"}
    return {"interference.measured": it, "interference.analytic": it, "hil": it,
            "tara": {"arch.sec"}, "review": {"item"}}.get(etype, set())

def trace_links(s):
    tl = {f"UT:{t}": [f"code.{t}"] for t in s["tasks"]}
    tl.update({f"SA:{t}": [f"code.{t}"] for t in s["tasks"]})
    tl["HIL"] = ["img.island"]; tl["E2E"] = ["code.e2e"]
    return tl

def claims(s):
    c = {f"TIM:{k}": [f"RTA:{k}", "INT"] + (["HIL"] if k == "I0" else []) for k in CORES}
    c["FFI"] = ["INT", "MPU"]
    c["SEC"] = ["TARA", "E2E"]
    for t in s["tasks"]:
        c[f"COR:{t}"] = [f"UT:{t}", f"SA:{t}"]
    c["TOP"] = ["HARA"]
    return c

# ----------------------------------------------------------------------------- changes
def change(s, kind, rng):
    s = copy.deepcopy(s); T = s["tasks"]
    isl = [t for t in T if T[t]["core"] in ISLAND and t not in ("BrakeCtl",)]
    app = [t for t in T if T[t]["core"] in APP]
    if kind == "realloc_island":
        t = rng.choice(isl); T[t]["core"] = "I0" if T[t]["core"] == "I1" else "I1"
        T[t]["crit"] = "D" if T[t]["core"] == "I0" else "B"
    elif kind in ("add_ids_island", "add_ids_app"):
        core = "I1" if kind == "add_ids_island" else rng.choice(APP)
        T["IDS"] = dict(core=core, C=round(rng.uniform(100, 300), 1), T=10000, r=2.5,
                        sram=True, crit="QM", code=0, behav=0, can=True)
        if core in ISLAND:
            end = max(b + z for b, z in s["parts"].values())
            s["parts"]["IDS"] = (end + 1024, 8192)
        s["controls"] = sorted(set(s["controls"]) | {"IDS"})
    elif kind == "realloc_app":
        t = rng.choice(app); T[t]["core"] = rng.choice([k for k in APP if k != T[t]["core"]])
    elif kind == "app_sram_enable":
        cand = [t for t in app if not T[t]["sram"]] or app
        t = rng.choice(cand); T[t]["sram"] = True
    elif kind == "wcet_change":
        t = rng.choice(isl + ["BrakeCtl"]); T[t]["C"] = round(T[t]["C"] * rng.uniform(1.1, 1.3), 1)
        T[t]["code"] += 1; T[t]["behav"] += 1
    elif kind == "refactor":
        t = rng.choice(isl + ["BrakeCtl"]); T[t]["code"] += 1
    elif kind == "period_change":
        t = rng.choice(isl); T[t]["T"] = rng.choice([p * 1000 for p in PERIODS_MS if p * 1000 != T[t]["T"] and p >= 2])
    elif kind == "ic_change":
        s["ic"]["L"] = round(s["ic"]["L"] * rng.choice([0.5, 1.5, 2.0]), 3)
    elif kind == "partition_resize":
        t = rng.choice(sorted(s["parts"])); b, z = s["parts"][t]; s["parts"][t] = (b, int(z * rng.uniform(1.1, 1.6)))
    elif kind == "can_add_msg":
        s["can"][f"msgN{rng.randint(0, 99)}"] = rng.random() < 0.5
    elif kind == "dma_enable":
        s["dma"] = True
    elif kind == "app_code_change":
        t = rng.choice(app); T[t]["C"] = round(T[t]["C"] * rng.uniform(1.1, 1.5), 1)
        T[t]["code"] += 1; T[t]["behav"] += 1
    return s

CHANGES = ["add_ids_island", "add_ids_app", "realloc_island", "realloc_app", "app_sram_enable",
           "wcet_change", "refactor", "period_change", "ic_change", "partition_resize",
           "can_add_msg", "app_code_change", "dma_enable"]

# ----------------------------------------------------------------------------- methods
def closure(flagged, ev):
    f = set(flagged); changed = True
    while changed:
        changed = False
        for e, d in ev.items():
            if e not in f and set(d["D"]) & f:
                f.add(e); changed = True
    return f

def flagged_sets(s0, s1, int_variant):
    out = {}
    for fl in (False, True):
        a0, a1 = artifacts(s0, fl), artifacts(s1, fl)
        C = {k for k in set(a0) | set(a1) if a0.get(k) != a1.get(k)}
        ev = evidence(s0, int_variant, fl)
        man = {e for e, d in ev.items() if set(d["M"]) & C}
        direct = {e for e, d in ev.items() if (set(d["M"]) | set(d["X"])) & C}
        if not fl:
            tl = {e for e, arts in trace_links(s0).items() if set(arts) & C}
            out["trace"] = tl; out["manifest"] = man
            out["aag_direct"] = direct; out["aag_closed"] = closure(direct, ev)
            out["_ev"] = ev
        else:
            out["aag_closed_file"] = closure(direct, ev)
    return out

def perturb_scopes(ev, s, rng, p_omit, p_add, checks, universe):
    """A1 proposes scopes = true scope with omissions/inventions; with checks, a proposal that
    misses a manifest or template entry is rejected and corrected by a human."""
    new, escal = {}, 0
    for e, d in ev.items():
        true = set(d["M"]) | set(d["X"])
        prop = {a for a in true if rng.random() >= p_omit}
        if rng.random() < p_add:
            prop.add(rng.choice(universe))
        if checks:
            need = set(d["M"]) | (template(s, d["type"]) & true)
            if not need <= prop:
                prop = true | (prop - true); escal += 1   # human fixes the omission
        new[e] = dict(d, S=prop)
    return new, escal

# ----------------------------------------------------------------------------- main
def run(n_sys=300, seed=1):
    rng = random.Random(seed)
    methods = ["trace", "manifest", "aag_direct", "aag_closed", "aag_cutoff", "aag_closed_file"]
    agg = {v: {c: defaultdict(float) for c in CHANGES + ["ALL"]} for v in ("analytic", "measured")}
    scope_err = defaultdict(lambda: defaultdict(float))
    p_grid = [0.0, 0.05, 0.1, 0.2, 0.3]
    for i in range(n_sys):
        s0 = gen_system(rng)
        for ck in CHANGES:
            s1 = change(s0, ck, rng)
            ev_ids = list(evidence(s0, "analytic").keys())
            o0, o1 = outputs(s0, ev_ids), outputs(s1, ev_ids)
            stale = {e for e in ev_ids if o0[e] != o1[e]}
            crit = {e for e in stale if critical(e, o0[e], o1[e])}
            cl = claims(s0)
            for variant in ("analytic", "measured"):
                fs = flagged_sets(s0, s1, variant)
                ev = fs.pop("_ev")
                # early cut-off: re-run upstream items first; a consumer is re-run only if an
                # upstream output actually changed (known once the upstream re-run is done)
                cut = set(fs["aag_direct"]); grow = True
                while grow:
                    grow = False
                    for e, d in ev.items():
                        if e not in cut and any(x in cut and o0[x] != o1[x] for x in d["D"]):
                            cut.add(e); grow = True
                fs["aag_cutoff"] = cut
                for bucket in (ck, "ALL"):
                    A = agg[variant][bucket]
                    A["n"] += 1; A["stale"] += len(stale); A["crit"] += len(crit)
                    A["any_stale"] += bool(stale)
                    A["need_h"] += sum(COST_H[ev[e]["type"]] for e in stale)
                    for m in methods:
                        f = fs[m]
                        A[f"{m}.tp"] += len(f & stale); A[f"{m}.fp"] += len(f - stale)
                        A[f"{m}.miss"] += len(stale - f); A[f"{m}.critmiss"] += len(crit - f)
                        A[f"{m}.scen_miss"] += bool(stale - f); A[f"{m}.scen_critmiss"] += bool(crit - f)
                        A[f"{m}.hours"] += sum(COST_H[ev[e]["type"]] for e in f if e in ev)
                        if bucket == "ALL":
                            for e in stale:
                                ty = ev[e]["type"].split(".")[0]
                                A[f"type.{ty}.stale"] += 1 if m == "trace" else 0
                                A[f"{m}.type.{ty}.hit"] += e in f
                        bad = [c for c, es in cl.items() if set(es) & stale and not set(es) & f]
                        A[f"{m}.false_sup"] += len(bad)
                        A[f"{m}.scen_false_sup"] += bool(bad)
            # scope-error sensitivity (measured interference variant: largest non-manifest scopes)
            ev = evidence(s0, "measured")
            a0, a1 = artifacts(s0), artifacts(s1)
            C = {k for k in set(a0) | set(a1) if a0.get(k) != a1.get(k)}
            universe = sorted(a0)
            for p in p_grid:
                for checks in (False, True):
                    pr, esc = perturb_scopes(ev, s0, rng, p, 0.05, checks, universe)
                    f = closure({e for e, d in pr.items() if d["S"] & C}, ev)
                    k = f"p={p}|checks={checks}"
                    S = scope_err[k]
                    S["n"] += 1; S["miss"] += len(stale - f); S["stale"] += len(stale)
                    S["critmiss"] += len(crit - f); S["scen_miss"] += bool(stale - f)
                    S["escalations"] += esc; S["n_ev"] += len(ev)
                    S["fp"] += len(f - stale)
    return agg, scope_err

def summarize(agg, scope_err):
    out = {"methods": {}, "scope_errors": {}}
    for variant, by in agg.items():
        out["methods"][variant] = {}
        for ck, A in by.items():
            row = {"scenarios": int(A["n"]), "stale_items": int(A["stale"]), "critical_items": int(A["crit"]),
                   "scenarios_with_stale": int(A["any_stale"]), "needed_hours": round(A["need_h"], 1)}
            for m in ["trace", "manifest", "aag_direct", "aag_closed", "aag_cutoff", "aag_closed_file"]:
                tp, fp, miss = A[f"{m}.tp"], A[f"{m}.fp"], A[f"{m}.miss"]
                row[m] = {"recall": round(tp / (tp + miss), 4) if tp + miss else None,
                          "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                          "missed_items": int(miss), "missed_critical": int(A[f"{m}.critmiss"]),
                          "scenarios_with_miss": int(A[f"{m}.scen_miss"]),
                          "scenarios_with_critical_miss": int(A[f"{m}.scen_critmiss"]),
                          "claims_falsely_supported": int(A[f"{m}.false_sup"]),
                          "scenarios_with_false_support": int(A[f"{m}.scen_false_sup"]),
                          "reverification_hours": round(A[f"{m}.hours"], 1)}
            if ck == "ALL":
                types = sorted({k.split(".")[1] for k in A if k.startswith("type.")})
                row["per_type"] = {ty: {"stale": int(A[f"type.{ty}.stale"]),
                                        **{m: round(A[f"{m}.type.{ty}.hit"] / A[f"type.{ty}.stale"], 4)
                                           for m in ["trace", "manifest", "aag_direct", "aag_closed", "aag_cutoff"]}}
                                   for ty in types}
            out["methods"][variant][ck] = row
    for k, S in sorted(scope_err.items()):
        out["scope_errors"][k] = {"recall": round(1 - S["miss"] / S["stale"], 4) if S["stale"] else None,
                                  "missed_items": int(S["miss"]), "missed_critical": int(S["critmiss"]),
                                  "scenarios_with_miss": int(S["scen_miss"]), "scenarios": int(S["n"]),
                                  "escalation_rate": round(S["escalations"] / S["n_ev"], 4),
                                  "false_positives": int(S["fp"])}
    return out

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    agg, se = run(n, seed)
    res = summarize(agg, se); res["config"] = {"n_systems": n, "seed": seed, "changes": CHANGES, "cost_hours": COST_H}
    json.dump(res, sys.stdout, indent=1)
