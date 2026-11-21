#!/usr/bin/env python3
"""Language-model agents for the Agentic Assurance Graph (AAG), and their evaluation.

Agents (Sec. "Bounded Agent Roles" of the paper) propose; the deterministic core decides.
  A1  evidence ingestion: reads a tool report or analysis record and proposes the evidence
      scope S(e) (artifact keys) and the evidence items it consumes D(e), with a rationale.
  A3  re-verification planning: reads an engineering change request, the graph and the core's
      invalidation set I(C); proposes an order, a causal rationale per item, and any extra items
      it believes need re-verification (extras can only add work, never remove it).
  IMPACT (baseline, not an AAG role): the same model is asked to decide on its own which
      evidence a change makes stale, with no core. This measures what would slip through if
      the model's judgement were trusted.

Scenarios are a sample of the 19,500 benchmark scenarios: the first K systems of bench.run for
seeds 1-3 (systems are i.i.d., so the first K are a random sample), with the exact changed
systems that bench.run produced. Ground truth comes from re-running the analyses (bench.outputs).

Everything except the model call is deterministic and runs offline. Offline proposers
("oracle", "parser+templates", "empty") exercise the full pipeline without a model.

Usage (offline):
  python3 agents.py tasks OUT_DIR K1 K3            # write a1_tasks.jsonl, a3_tasks.jsonl
  python3 agents.py offline OUT_DIR PROPOSER       # answer all tasks without a model
  python3 agents.py score OUT_DIR ANSWERS.jsonl    # write score JSON
The Colab notebook (../notebooks/run_agents_colab.ipynb) calls the same functions and sends the
prompts to the Anthropic Message Batches API.
"""
import copy, json, os, random, re, sys
from collections import defaultdict
import bench
import a1_eval

MODEL = "claude-opus-5-5"          # pinned for reproducibility
EFFORT = "medium"                  # explicit (also the model's default)
PRICE_IN, PRICE_OUT = 4.0, 20.0    # USD per million tokens, standard; batch is 50%
BATCH_DISCOUNT = 0.5
MODEL_ITEMS = ("INT", "RTA", "MPU", "HIL", "E2E", "TARA", "HARA")   # A1 runs on these kinds
INT_VARIANT = "measured"           # largest non-manifest scopes (as in RQ4)

# ============================================================================ scenarios
def sample_scenarios(k, seeds=(1, 2, 3)):
    """First k systems of bench.run(., seed) for each seed, with the 13 changed systems that
    bench.run itself generated (same random stream), i.e. exact benchmark scenarios."""
    out = []
    for seed in seeds:
        rec = []
        orig = bench.change
        def spy(s, kind, rng, _o=orig):
            s1 = _o(s, kind, rng); rec.append((copy.deepcopy(s), kind, copy.deepcopy(s1))); return s1
        bench.change = spy
        try:
            bench.run(k, seed)
        finally:
            bench.change = orig
        for j in range(k):
            chunk = rec[j * len(bench.CHANGES):(j + 1) * len(bench.CHANGES)]
            out.append(dict(sid=f"seed{seed}.sys{j}", seed=seed, idx=j, s0=chunk[0][0],
                            changes=[(ck, s1) for _, ck, s1 in chunk]))
    return out

def truth_for(s0, s1):
    ev_ids = list(bench.evidence(s0, INT_VARIANT).keys())
    o0, o1 = bench.outputs(s0, ev_ids), bench.outputs(s1, ev_ids)
    stale = {e for e in ev_ids if o0[e] != o1[e]}
    crit = {e for e in stale if bench.critical(e, o0[e], o1[e])}
    a0, a1 = bench.artifacts(s0), bench.artifacts(s1)
    C = {k for k in set(a0) | set(a1) if a0.get(k) != a1.get(k)}
    return stale, crit, C

def core_flag(ev, C, scopes=None, consumes=None):
    """AAG core: direct invalidation by scope intersection, closed over consumes."""
    S = scopes or {e: set(d["M"]) | set(d["X"]) for e, d in ev.items()}
    D = consumes or {e: set(d["D"]) for e, d in ev.items()}
    f = {e for e in ev if S[e] & C}
    grow = True
    while grow:
        grow = False
        for e in ev:
            if e not in f and D[e] & f:
                f.add(e); grow = True
    return f

def check_proposal(manifest, true_scope, etype, template, scope, consumes):
    """Deterministic scope checks (Sec. Scope Completeness): manifest containment, type
    template containment, and the consumes edge every RTA has to the interference result.
    Returns the list of violated requirements (empty = accepted)."""
    need = set(manifest) | (set(template) & set(true_scope))
    v = sorted(need - set(scope))
    if etype == "rta" and "INT" not in consumes:
        v.append("consumes:INT")
    return v

# ============================================================================ text rendering
def describe_change(s0, s1, ck):
    """An engineering change request (ECR) as an engineer would write it. Uses artifact-level
    names (task and core ids) but never lists which evidence is affected."""
    T0, T1 = s0["tasks"], s1["tasks"]
    moved = [t for t in T0 if t in T1 and T0[t]["core"] != T1[t]["core"]]
    if ck in ("add_ids_island", "add_ids_app"):
        x = T1["IDS"]
        txt = (f"Add a CAN intrusion-detection task IDS on core {x['core']}: period 10 ms, WCET {x['C']} us, "
               f"reads every CAN frame, keeps its signature tables in the shared island SRAM. "
               f"Security architecture updated to list IDS as a control.")
        if "IDS" in s1["parts"]:
            txt += " A new MPU partition is created for IDS."
        return txt
    if ck in ("realloc_island", "realloc_app"):
        t = moved[0]
        return f"Load balancing: move task {t} from core {T0[t]['core']} to core {T1[t]['core']}. No code change."
    if ck == "app_sram_enable":
        t = [t for t in T0 if T0[t]["sram"] != T1[t]["sram"]]
        t = t[0] if t else sorted(t for t in T0 if T0[t]["core"] in bench.APP)[0]
        return f"Task {t} (core {T0[t]['core']}) moves its working buffers from local RAM to the shared island SRAM. No code change."
    if ck in ("wcet_change", "app_code_change"):
        t = [t for t in T0 if T0[t]["C"] != T1[t]["C"]][0]
        return (f"Feature change in task {t} (core {T0[t]['core']}): new filtering logic; source code and unit "
                f"behaviour change; measured WCET grows from {T0[t]['C']} us to {T1[t]['C']} us.")
    if ck == "refactor":
        t = [t for t in T0 if T0[t]["code"] != T1[t]["code"]][0]
        return f"Refactoring of task {t} (core {T0[t]['core']}): source restructured, behaviour and WCET unchanged."
    if ck == "period_change":
        t = [t for t in T0 if T0[t]["T"] != T1[t]["T"]][0]
        return f"Period of task {t} (core {T0[t]['core']}) changes from {T0[t]['T']//1000} ms to {T1[t]['T']//1000} ms."
    if ck == "ic_change":
        return (f"Interconnect configuration: SRAM port arbitration slot latency changes from "
                f"{int(s0['ic']['L']*1000)} ns to {int(s1['ic']['L']*1000)} ns.")
    if ck == "partition_resize":
        t = [t for t in s0["parts"] if s0["parts"][t] != s1["parts"][t]][0]
        return f"MPU partition of task {t} grows from {s0['parts'][t][1]} to {s1['parts'][t][1]} bytes (same base address)."
    if ck == "can_add_msg":
        m = [m for m in s1["can"] if m not in s0["can"]]
        m = m[0] if m else sorted(s1["can"])[-1]
        return f"Add CAN message {m} to the communication matrix ({'E2E protected' if s1['can'][m] else 'not E2E protected'})."
    if ck == "dma_enable":
        return "Enable the application-cluster DMA engine to copy camera frames into a staging buffer in the shared island SRAM."
    raise KeyError(ck)

def render_graph(s, ev):
    lines = []
    for e in sorted(ev):
        d = ev[e]
        sc = sorted(set(d["M"]) | set(d["X"]))
        lines.append(f"{e} [{d['type']}] scope: {', '.join(sc) or '-'}" + (f"; consumes: {', '.join(d['D'])}" if d["D"] else ""))
    return "\n".join(lines)

def render_catalogue(s):
    return "\n".join(f"{k}: {v}" for k, v in a1_eval.catalogue(s).items())

# ============================================================================ prompts + schemas
SYSTEM = ("You are an assistant inside an assurance-maintenance tool for a safety-critical multicore "
          "automotive ECU (two lockstep safety-island cores I0, I1 and four application cores A0-A3 sharing an "
          "island SRAM port). Your output is a proposal that deterministic checks and a human engineer will review; "
          "you never decide whether a claim holds. Be complete and precise. Answer only in the requested JSON.")

A1_PROMPT = """Task: evidence ingestion. A tool report or analysis record is given below. Propose its evidence scope:
every configuration artifact (from the catalogue) whose change could change this result, and every other evidence
item whose result this one consumes. Include dependencies that the report does not name explicitly when the kind of
analysis depends on them (for example, multicore interference depends on what runs on the other cores that reach the
shared SRAM port). Do not include artifacts the result cannot depend on. Give a short rationale.

Evidence id: {eid}
Report:
<<<
{report}
>>>

Artifact catalogue (key: description):
{cat}

Other evidence ids: {evs}"""

A1_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["scope", "consumes", "rationale"],
             "properties": {"scope": {"type": "array", "items": {"type": "string"}},
                            "consumes": {"type": "array", "items": {"type": "string"}},
                            "rationale": {"type": "string"}}}

IMPACT_PROMPT = """Task: change-impact analysis. An engineering change request (ECR) is given, together with the assurance
graph: every evidence item with its type, its scope (the artifacts its result depends on) and the evidence results it
consumes. Decide which evidence items this change makes stale, that is, whose result could differ if the analysis or
test were re-run on the changed system. For each, give the reason in one sentence.

ECR: {ecr}

Artifact catalogue (key: description):
{cat}

Assurance graph:
{graph}"""

IMPACT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["stale"],
                 "properties": {"stale": {"type": "array", "items": {
                     "type": "object", "additionalProperties": False, "required": ["evidence", "reason"],
                     "properties": {"evidence": {"type": "string"}, "reason": {"type": "string"}}}}}}

A3_PROMPT = """Task: re-verification planning. An engineering change request (ECR) is given, together with the assurance
graph and the invalidation set I(C) that the deterministic core computed (items whose scope contains a changed
artifact, closed over consumed results). Propose a re-verification plan:
- "order": every item of I(C) exactly once, in the order to re-run them. An item must come after every item of I(C)
  it consumes. Among independent items, put cheap ones first.
- "items": for each item of I(C), the causal path from the change to it, as a list that starts with one changed artifact
  key from the item's scope, or with an item of I(C) that this item consumes, and ends with the item id; plus a
  one-sentence rationale.
- "extra": evidence items NOT in I(C) that you believe the change could still make stale (for example through a
  dependency no scope records), each with a reason. Leave it empty if there are none. Extras only add work.

Changed artifact keys C (from content hashes): {changed}
I(C): {inv}

ECR: {ecr}

Artifact catalogue (key: description):
{cat}

Assurance graph:
{graph}"""

A3_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["order", "items", "extra"],
             "properties": {
                 "order": {"type": "array", "items": {"type": "string"}},
                 "items": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                           "required": ["evidence", "path", "rationale"],
                           "properties": {"evidence": {"type": "string"},
                                          "path": {"type": "array", "items": {"type": "string"}},
                                          "rationale": {"type": "string"}}}},
                 "extra": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                           "required": ["evidence", "reason"],
                           "properties": {"evidence": {"type": "string"}, "reason": {"type": "string"}}}}}}

SCHEMAS = {"a1": A1_SCHEMA, "impact": IMPACT_SCHEMA, "a3": A3_SCHEMA}
MAX_TOKENS = {"a1": 16000, "impact": 24000, "a3": 32000}

# ============================================================================ task generation
def gen_tasks(k1, k3, seed=7):
    """A1 tasks on the first k1 systems per seed, IMPACT and A3 tasks on the first k3 systems
    per seed (all 13 changes). Report styles are drawn with a fixed seed."""
    rng = random.Random(seed)
    scen = sample_scenarios(max(k1, k3))
    a1, a3 = [], []
    for sc in scen:
        s0 = sc["s0"]; ev = bench.evidence(s0, INT_VARIANT); cat = render_catalogue(s0)
        if sc["idx"] < k1:
            for e in sorted(ev):
                d = ev[e]; kind = e.split(":")[0]
                style = rng.choice(a1_eval.STYLES) if kind in MODEL_ITEMS else "structured"
                report = a1_eval.render(s0, e, d, style, rng)
                t = dict(id=f"a1|{sc['sid']}|{e}", agent="a1", sid=sc["sid"], evidence=e, type=d["type"], style=style,
                         model=kind in MODEL_ITEMS, report=report,
                         truth_scope=sorted(set(d["M"]) | set(d["X"])), truth_consumes=d["D"],
                         manifest=d["M"], template=sorted(bench.template(s0, d["type"])),
                         catalogue_keys=sorted(a1_eval.catalogue(s0)), evidence_ids=sorted(ev))
                if t["model"]:
                    t["prompt"] = A1_PROMPT.format(eid=e, report=report, cat=cat,
                                                  evs=", ".join(x for x in sorted(ev) if x != e))
                a1.append(t)
        if sc["idx"] < k3:
            graph = render_graph(s0, ev)
            for ck, s1 in sc["changes"]:
                stale, crit, C = truth_for(s0, s1)
                inv = core_flag(ev, C)
                ecr = describe_change(s0, s1, ck)
                base = dict(sid=sc["sid"], change=ck, ecr=ecr, stale=sorted(stale), crit=sorted(crit),
                            changed=sorted(C), inv=sorted(inv), evidence_ids=sorted(ev),
                            consumes={e: ev[e]["D"] for e in ev},
                            scope={e: sorted(set(ev[e]["M"]) | set(ev[e]["X"])) for e in ev},
                            cost={e: bench.COST_H[ev[e]["type"]] for e in ev})
                a3.append(dict(base, id=f"impact|{sc['sid']}|{ck}", agent="impact",
                               prompt=IMPACT_PROMPT.format(ecr=ecr, cat=cat, graph=graph)))
                a3.append(dict(base, id=f"a3|{sc['sid']}|{ck}", agent="a3",
                               prompt=A3_PROMPT.format(ecr=ecr, cat=cat, graph=graph, changed=", ".join(sorted(C)) or "(none)",
                                                       inv=", ".join(sorted(inv)) or "(empty)")))
    return a1, a3

def request_params(task, model=MODEL, effort=EFFORT):
    """Messages API parameters for one task (used for both batch and direct calls)."""
    return dict(model=model, max_tokens=MAX_TOKENS[task["agent"]], system=SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMAS[task["agent"]]}},
                messages=[{"role": "user", "content": task["prompt"]}])

def cost_usd(usage, batch=True):
    f = BATCH_DISCOUNT if batch else 1.0
    i = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) * 1.25 + \
        usage.get("cache_read_input_tokens", 0) * 0.1
    return f * (i * PRICE_IN + usage.get("output_tokens", 0) * PRICE_OUT) / 1e6

# ============================================================================ offline proposers
def offline_answer(task, proposer, rng):
    a = task["agent"]
    if a == "a1":
        if proposer == "oracle":
            return {"scope": task["truth_scope"], "consumes": task["truth_consumes"], "rationale": "oracle"}
        if proposer == "parser+templates":
            p = a1_eval.parser(dict(task, catalogue={k: "" for k in task["catalogue_keys"]}))
            p["scope"] = sorted(set(p["scope"]) | set(task["template"]))
            if task["type"] == "rta": p["consumes"] = sorted(set(p["consumes"]) | {"INT"})
            return dict(p, rationale="parser+templates")
        if proposer == "noisy":   # oracle with 20% omissions, for testing the checks
            return {"scope": [x for x in task["truth_scope"] if rng.random() > 0.2],
                    "consumes": task["truth_consumes"], "rationale": "noisy"}
        return {"scope": [], "consumes": [], "rationale": "empty"}
    if a == "impact":
        st = task["inv"] if proposer == "oracle" else []
        return {"stale": [{"evidence": e, "reason": proposer} for e in st]}
    if a == "a3":
        order = topo(task["inv"], task["consumes"], task["cost"])
        items = []
        for e in order:
            first = sorted(set(task["scope"][e]) & set(task["changed"])) or \
                    sorted(set(task["consumes"][e]) & set(task["inv"]))
            items.append({"evidence": e, "path": first[:1] + [e], "rationale": proposer})
        return {"order": order, "items": items, "extra": []}

def topo(items, consumes, cost):
    left, out = set(items), []
    while left:
        ready = sorted([e for e in left if not (set(consumes[e]) & left)], key=lambda e: (cost[e], e))
        if not ready:
            ready = sorted(left)
        out.append(ready[0]); left.discard(ready[0])
    return out

def a1_fill(task):
    """A1 is called only for reports without a machine-readable manifest. Structured unit-test
    and static-analysis reports name their unit literally; the parser handles them."""
    p = a1_eval.parser(dict(task, catalogue={k: "" for k in task["catalogue_keys"]}))
    return dict(p, rationale="parser (structured report)")

# ============================================================================ scoring
def parse_answer(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except Exception:
            return None

def score(a1_tasks, a3_tasks, answers, scen=None):
    """answers: id -> parsed answer dict (None = refusal/error/unparseable)."""
    R = {}
    # ---------------------------------------------------------------- A1 proposal quality
    by = defaultdict(lambda: defaultdict(float))
    props = defaultdict(dict)       # sid -> e -> (scope, consumes, rejected)
    for t in a1_tasks:
        ans = answers.get(t["id"]) if t["model"] else a1_fill(t)
        failed = ans is None
        ans = ans or {"scope": [], "consumes": []}
        sc, co = set(map(str, ans.get("scope", []))), set(map(str, ans.get("consumes", [])))
        true, tc = set(t["truth_scope"]), set(t["truth_consumes"])
        viol = check_proposal(t["manifest"], t["truth_scope"], t["type"], t["template"], sc, co) \
            if not failed else ["no-answer"]
        props[t["sid"]][t["evidence"]] = (sc, co, bool(viol))
        if not t["model"]:
            continue
        omitted, invented = true - sc, sc - true
        unknown = sc - set(t["catalogue_keys"])
        bad = bool(omitted) or bool(tc - co)
        for key in ("ALL", f"type:{t['type']}", f"style:{t['style']}"):
            b = by[key]
            b["n"] += 1; b["true"] += len(true); b["omitted"] += len(omitted); b["invented"] += len(invented)
            b["unknown_keys"] += len(unknown); b["exact"] += (not omitted and not invented and tc == co)
            b["with_omission"] += bad; b["rejected"] += bool(viol); b["bad_rejected"] += bad and bool(viol)
            b["bad_accepted"] += bad and not viol; b["good_rejected"] += (not bad) and bool(viol)
            b["failed"] += failed; b["consume_missed"] += len(tc - co); b["consume_invented"] += len(co - tc)
    for b in by.values():
        n = b["n"]
        b["omission_rate"] = b["omitted"] / b["true"] if b["true"] else None
        b["exact_rate"] = b["exact"] / n
        b["items_with_omission_rate"] = b["with_omission"] / n
        b["rejection_rate"] = b["rejected"] / n
        b["bad_rejected_share"] = b["bad_rejected"] / b["with_omission"] if b["with_omission"] else None
        b["undetected_omission_rate"] = b["bad_accepted"] / n
    R["a1"] = {k: dict(v) for k, v in sorted(by.items())}

    # ---------------------------------------------------------------- A1 end-to-end (stale evidence)
    if scen:
        E = defaultdict(float)
        for sc_ in scen:
            sid = sc_["sid"]
            if sid not in props:
                continue
            s0 = sc_["s0"]; ev = bench.evidence(s0, INT_VARIANT)
            P = props[sid]
            for mode in ("true", "agent_nochecks", "agent_checks"):
                if mode == "true":
                    S = D = None
                else:
                    S, D = {}, {}
                    for e, d in ev.items():
                        sc, co, rej = P[e]
                        if mode == "agent_checks" and rej:
                            # a human corrects a rejected proposal by adding what it omitted; the
                            # agent's additional entries stay (as in bench.perturb_scopes)
                            sc, co = sc | set(d["M"]) | set(d["X"]), co | set(d["D"])
                        S[e], D[e] = sc, co
                for ck, s1 in sc_["changes"]:
                    stale, crit, C = truth_for(s0, s1)
                    f = core_flag(ev, C, S, D)
                    E[f"{mode}.n"] += 1; E[f"{mode}.stale"] += len(stale); E[f"{mode}.miss"] += len(stale - f)
                    E[f"{mode}.critmiss"] += len(crit - f); E[f"{mode}.scen_critmiss"] += bool(crit - f)
                    E[f"{mode}.scen_miss"] += bool(stale - f); E[f"{mode}.fp"] += len(f - stale)
                    E[f"{mode}.hours"] += sum(bench.COST_H[ev[e]["type"]] for e in f)
                    E[f"{mode}.need_h"] += sum(bench.COST_H[ev[e]["type"]] for e in stale)
                    if ck != "dma_enable":
                        E[f"{mode}.nodma_stale"] += len(stale); E[f"{mode}.nodma_miss"] += len(stale - f)
            E["systems"] += 1
            E["escalated"] += sum(1 for e in ev if P[e][2]); E["items"] += len(ev)
        for m in ("true", "agent_nochecks", "agent_checks"):
            if E[f"{m}.stale"]:
                E[f"{m}.recall"] = 1 - E[f"{m}.miss"] / E[f"{m}.stale"]
                E[f"{m}.recall_nodma"] = 1 - E[f"{m}.nodma_miss"] / E[f"{m}.nodma_stale"]
                E[f"{m}.effort_ratio"] = E[f"{m}.hours"] / E[f"{m}.need_h"]
        R["a1_end_to_end"] = dict(E)

    # ---------------------------------------------------------------- IMPACT (model alone) and A3 (with core)
    I = defaultdict(lambda: defaultdict(float))
    A = defaultdict(lambda: defaultdict(float))
    for t in a3_tasks:
        ans = answers.get(t["id"])
        stale, crit, inv = set(t["stale"]), set(t["crit"]), set(t["inv"])
        evs = set(t["evidence_ids"])
        if t["agent"] == "impact":
            for key in ("ALL", f"change:{t['change']}"):
                b = I[key]; b["n"] += 1; b["failed"] += ans is None
                pred = {x.get("evidence") for x in (ans or {}).get("stale", [])} & evs
                b["stale"] += len(stale); b["tp"] += len(pred & stale); b["fp"] += len(pred - stale)
                b["miss"] += len(stale - pred); b["crit"] += len(crit); b["critmiss"] += len(crit - pred)
                b["scen_miss"] += bool(stale - pred); b["scen_critmiss"] += bool(crit - pred)
                b["unknown_ids"] += len({x.get("evidence") for x in (ans or {}).get("stale", [])} - evs)
                b["core_miss"] += len(stale - inv); b["core_critmiss"] += len(crit - inv)
                b["hours"] += sum(t["cost"][e] for e in pred); b["need_h"] += sum(t["cost"][e] for e in stale)
                b["core_hours"] += sum(t["cost"][e] for e in inv)
        else:
            ans = ans or {}
            order = [str(x) for x in ans.get("order", [])]
            pos = {e: i for i, e in enumerate(order)}
            complete = set(order) == inv and len(order) == len(inv)
            dep_ok = all(pos[d] < pos[e] for e in order if e in inv for d in t["consumes"].get(e, []) if d in inv and d in pos)
            valid = complete and dep_ok and bool(ans)
            items = {x.get("evidence"): x for x in ans.get("items", [])}
            path_ok = path_n = 0
            for e in inv:
                x = items.get(e); path_n += 1
                if not x or not x.get("path"):
                    continue
                p0 = x["path"][0]
                if (p0 in t["changed"] and p0 in t["scope"][e]) or (p0 in inv and p0 in t["consumes"][e]):
                    path_ok += 1
            extra = {x.get("evidence") for x in ans.get("extra", [])} & evs
            missed_by_core = stale - inv
            for key in ("ALL", f"change:{t['change']}"):
                b = A[key]; b["n"] += 1; b["failed"] += not ans
                b["plan_valid"] += valid; b["plan_incomplete"] += (not complete) and bool(ans)
                b["plan_dep_violation"] += (not dep_ok) and bool(ans)
                b["path_ok"] += path_ok; b["path_n"] += path_n
                b["extra"] += len(extra); b["extra_stale"] += len(extra & missed_by_core)
                b["extra_redundant"] += len(extra & inv)
                b["core_miss"] += len(missed_by_core); b["core_critmiss"] += len(crit - inv)
                b["union_miss"] += len(missed_by_core - extra); b["union_critmiss"] += len(crit - inv - extra)
                b["extra_hours"] += sum(t["cost"][e] for e in extra - inv)
                b["core_hours"] += sum(t["cost"][e] for e in inv); b["stale"] += len(stale)
                b["need_h"] += sum(t["cost"][e] for e in stale)
    for b in I.values():
        b["recall"] = b["tp"] / b["stale"] if b["stale"] else None
        b["precision"] = b["tp"] / (b["tp"] + b["fp"]) if (b["tp"] + b["fp"]) else None
        b["core_recall"] = 1 - b["core_miss"] / b["stale"] if b["stale"] else None
        b["effort_ratio"] = b["hours"] / b["need_h"] if b["need_h"] else None
    for b in A.values():
        b["plan_valid_rate"] = b["plan_valid"] / b["n"]
        b["path_ok_rate"] = b["path_ok"] / b["path_n"] if b["path_n"] else None
        b["core_recall"] = 1 - b["core_miss"] / b["stale"] if b["stale"] else None
        b["union_recall"] = 1 - b["union_miss"] / b["stale"] if b["stale"] else None
    R["impact"] = {k: dict(v) for k, v in sorted(I.items())}
    R["a3"] = {k: dict(v) for k, v in sorted(A.items())}
    return R

# ============================================================================ CLI
def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def _read(path):
    return [json.loads(l) for l in open(path)]

if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    if cmd == "tasks":
        a1, a3 = gen_tasks(int(sys.argv[3]), int(sys.argv[4]))
        _write(f"{out}/a1_tasks.jsonl", a1); _write(f"{out}/a3_tasks.jsonl", a3)
        n = sum(t["model"] for t in a1) + len(a3)
        chars = sum(len(t.get("prompt", "")) for t in a1 + a3)
        print(json.dumps({"a1_tasks": len(a1), "a1_model_calls": sum(t["model"] for t in a1),
                          "a3_impact_calls": len(a3), "model_calls": n, "prompt_chars": chars,
                          "approx_input_tokens": chars // 4}))
    elif cmd == "offline":
        prop = sys.argv[3]; rng = random.Random(0)
        rows = [{"id": t["id"], "answer": offline_answer(t, prop, rng)}
                for t in _read(f"{out}/a1_tasks.jsonl") + _read(f"{out}/a3_tasks.jsonl") if t.get("prompt")]
        _write(f"{out}/answers_{prop}.jsonl", rows)
    elif cmd == "score":
        a1, a3 = _read(f"{out}/a1_tasks.jsonl"), _read(f"{out}/a3_tasks.jsonl")
        ans = {r["id"]: r["answer"] for r in _read(sys.argv[3])}
        k = max(int(t["sid"].split("sys")[1]) for t in a1 + a3) + 1
        json.dump(score(a1, a3, ans, sample_scenarios(k)), sys.stdout, indent=1, default=float)
