#!/usr/bin/env python3
"""Evaluate agent A1 (evidence ingestion): from a tool report, propose the evidence scope S(e)
and consumed evidence D(e).

Reports are generated from benchmark systems (bench.py), so the true scope is known. Each
evidence item is rendered in one of three styles:
  structured : tool-native output that names artifact keys literally;
  vendor     : a different tool's output using aliases (core and task names differ from keys);
  prose      : an engineer's notes / review minutes in natural language.
Proposers:
  parser            : regex for literal artifact keys and evidence ids (no model);
  parser+templates  : parser output plus the type template (deterministic, no model);
  claude:<model>    : Anthropic API (needs ANTHROPIC_API_KEY);
  openai:<model>    : any OpenAI-compatible endpoint, e.g. a local open-weight model
                      (needs OPENAI_BASE_URL, optional OPENAI_API_KEY).
Usage:
  python3 a1_eval.py gen  N_SYSTEMS SEED out.jsonl         # generate tasks
  python3 a1_eval.py run  tasks.jsonl PROPOSER out.jsonl    # propose scopes
  python3 a1_eval.py score tasks.jsonl proposals.jsonl      # per-type error rates (JSON)
"""
import json, os, random, re, sys, time, urllib.request
import bench

STYLES = ["structured", "vendor", "prose"]

# ----------------------------------------------------------------------------- catalogue
def catalogue(s):
    """Artifact keys of a system with human-readable descriptions (what A1 sees)."""
    T = s["tasks"]; cat = {}
    for k in bench.CORES:
        cat[f"alloc.{k}"] = f"allocation row of core {k} ({'safety island' if k in bench.ISLAND else 'application cluster'})"
    for t, x in sorted(T.items()):
        cat[f"tim.{t}"] = f"timing parameters (WCET, period) of task {t}"
        cat[f"mem.{t}"] = f"memory-access parameters (SRAM use, access rate) of task {t}"
        cat[f"code.{t}"] = f"source code unit of task {t}"
    for t in sorted(s["parts"]):
        cat[f"part.{t}"] = f"MPU partition entry of task {t}"
    cat["part.index"] = "index of the MPU partition table (set of entries)"
    cat.update({"ic": "interconnect configuration (SRAM port arbitration, slot latency, which clusters are wired to the port)",
                "dma": "application-cluster DMA engine configuration",
                "arch.sec": "security-control architecture view (controls, tasks with CAN access)",
                "can": "CAN communication matrix (messages, E2E protection)",
                "code.e2e": "end-to-end protection library code", "rig": "HIL test rig configuration",
                "item": "item definition", "img.island": "built software image of the safety island",
                "img.app": "built software image of the application cluster",
                "report.INT": "interference analysis report file"})
    return cat

def aliases(rng):
    core = {"I0": "SafetyCore0", "I1": "SafetyCore1", "A0": "AppCPU0", "A1": "AppCPU1", "A2": "AppCPU2", "A3": "AppCPU3"}
    return core

def tname(t, style, core_alias):
    if style != "vendor":
        return t
    if t in ("BrakeCtl", "WSpeed", "IDS"):
        return {"BrakeCtl": "Brake_Control_5ms", "WSpeed": "WheelSpeed_Plaus_5ms", "IDS": "CAN_IDS"}[t]
    core, _, i = t.partition("t")
    return f"{core_alias[core]}_Runnable{i}"

# ----------------------------------------------------------------------------- reports
def render(s, e, d, style, rng):
    T = s["tasks"]; ca = aliases(rng); n = lambda t: tname(t, style, ca)
    kind, _, arg = e.partition(":")
    port_cores = bench.CORES if s["ic"]["app_port"] else bench.ISLAND
    L = s["ic"]["L"]
    if kind == "RTA":
        ts = sorted([t for t in T if T[t]["core"] == arg], key=lambda t: T[t]["T"])
        if style == "structured":
            rows = "\n".join(f"  {t}: C={T[t]['C']}us T={T[t]['T']}us inputs=[tim.{t}, mem.{t}]" for t in ts)
            return f"RTA-Tool v3.2 report\ncore: {arg}\ninput allocation: alloc.{arg}\ninterference input: report.INT (from evidence INT)\ntasks:\n{rows}\nresult: all deadlines met"
        if style == "vendor":
            rows = "\n".join(f"| {n(t)} | {T[t]['C']} | {T[t]['T']} | {T[t]['r']} acc/us |" for t in ts)
            return f"SchedAnalyzer Pro - Response Time Summary\nTarget: {ca[arg]}\nMulticore interference bounds imported from: interference_bounds.xml\n| Runnable | WCET(us) | Period(us) | Mem.intensity |\n{rows}"
        return (f"Notes from the timing review for {ca[arg]}: we re-ran the response-time analysis for the {len(ts)} "
                f"tasks mapped to that core ({', '.join(n(t) for t in ts)}), using the execution budgets and memory profiles from "
                f"the task configuration and the inflated execution times coming out of last week's interference analysis.")
    if kind == "INT":
        if style == "structured":
            return (f"IntfMeasure v1.4 log\nimages: img.island, img.app\nrig: rig\nmeasured worst per-access delay on island cores; "
                    f"co-runner load generated on {', '.join(port_cores)}")
        if style == "vendor":
            return (f"Multicore Interference Campaign Report\nBuilds under test: island-fw-build-{rng.randint(100,999)}, app-fw-build-{rng.randint(100,999)}\n"
                    f"Bench: HIL-B2\nScenario: all co-runner contenders active on {', '.join(ca[c] for c in port_cores)}; "
                    f"SRAM port slot setting {int(L*1000)} ns")
        return ("We measured the shared-SRAM interference on target with every other core running its normal "
                "load plus stress kernels. The bound holds for the current task-to-core mapping and the current interconnect arbitration settings.")
    if kind == "MPU":
        ents = sorted(s["parts"])
        if style == "structured":
            return "MPU-Check report\nentries checked: " + ", ".join(f"part.{t}" for t in ents) + "\nindex: part.index\nresult: no overlap"
        if style == "vendor":
            return "MemProt Verifier\nRegions: " + "; ".join(f"{n(t)}@0x{s['parts'][t][0]:06x}+{s['parts'][t][1]}" for t in ents)
        return f"Reviewed the MPU region table ({len(ents)} regions) for overlaps between partitions of different criticality; none found."
    if kind == "HIL":
        if style == "structured":
            return "HIL TC-17 log\nimage: img.island\nrig: rig\nmeasured brake latency within budget"
        if style == "vendor":
            return f"TestStand run TC-17 Brake_Latency\nDUT build island-fw-build-{rng.randint(100,999)} on HIL-B2\nmax latency 2.41 ms (n=500)"
        return ("Ran the brake pedal-to-command latency test on the rig with the current island firmware; the application cluster was "
                "running its production load at the same time, so the result reflects shared-memory contention from the other cores.")
    if kind in ("UT", "SA"):
        what = "unit tests" if kind == "UT" else "static analysis"
        if style == "structured":
            return f"{what} report\nunit: code.{arg}\nresult: pass"
        if style == "vendor":
            return f"{'VectorCAST' if kind == 'UT' else 'Polyspace'} summary for module {n(arg)}: 0 failures"
        return f"Completed {what} of the {n(arg)} software component; all findings dispositioned."
    if kind == "E2E":
        msgs = [m for m, p in sorted(s["can"].items()) if p]
        if style == "structured":
            return "E2E test report\nlibrary: code.e2e\nmatrix: can\nmessages: " + ", ".join(msgs)
        if style == "vendor":
            return "E2E Profile 5 conformance: " + ", ".join(msgs) + " -- PASS (lib E2E_P05 v2.1)"
        return "Checked end-to-end protection for all protected messages in the CAN database using the E2E library; all pass."
    if kind == "TARA":
        rows = "; ".join(m for m in sorted(s["can"]))
        if style == "structured":
            return f"TARA sheet\narchitecture: arch.sec\nmessages: can ({rows})\ncontrols: {', '.join(s['controls'])}"
        if style == "vendor":
            return f"Threat analysis workbook rev C. Assets: {rows}. Controls in place: {', '.join(s['controls'])}."
        return ("Threat analysis workshop for the CAN interface: we went through each message in the communication matrix "
                "and rated attack feasibility against the security controls shown in the current security architecture.")
    if kind == "HARA":
        return {"structured": "HARA review\nitem: item", "vendor": "Hazard analysis review minutes, item definition rev 4",
                "prose": "Reviewed the hazard analysis against the current item definition."}[style]
    raise KeyError(e)

def truth(d):
    return sorted(set(d["M"]) | set(d["X"]))

def gen(n_sys, seed, out):
    rng = random.Random(seed)
    with open(out, "w") as f:
        for i in range(n_sys):
            s = bench.gen_system(rng)
            ev = bench.evidence(s, "measured")
            # one item per evidence type per system keeps the mix balanced
            picks = ["INT", "MPU", "HIL", "E2E", "TARA", "HARA", f"RTA:{rng.choice(bench.CORES)}",
                     f"UT:{rng.choice(sorted(s['tasks']))}", f"SA:{rng.choice(sorted(s['tasks']))}"]
            cat = catalogue(s)
            for e in picks:
                style = rng.choice(STYLES)
                d = ev[e]
                f.write(json.dumps({"id": f"s{i}:{e}", "system": i, "seed": seed, "evidence": e, "type": d["type"],
                                    "style": style, "report": render(s, e, d, style, rng),
                                    "catalogue": cat, "evidence_ids": sorted(ev), "truth_scope": truth(d),
                                    "truth_consumes": d["D"], "manifest": d["M"],
                                    "template": sorted(bench.template(s, d["type"]))}) + "\n")

# ----------------------------------------------------------------------------- proposers
PROMPT = """You maintain an assurance graph for a safety-critical multicore ECU. A tool report or analysis record is given below.
Propose its evidence scope: every configuration artifact (from the catalogue) whose change could change this result, and every
other evidence item whose result this one consumes. Include dependencies that the report does not name explicitly if the kind of
analysis depends on them (e.g., multicore interference depends on what runs on the other cores). Do not include artifacts the
result cannot depend on.

Evidence id: {eid}
Report:
<<<
{report}
>>>

Artifact catalogue (key: description):
{cat}

Other evidence ids: {evs}

Answer with JSON only: {{"scope": [artifact keys], "consumes": [evidence ids]}}"""

def parser(task):
    txt = task["report"]
    keys = [k for k in task["catalogue"] if re.search(r"(?<![\w.])" + re.escape(k) + r"(?![\w])", txt)]
    cons = [e for e in task["evidence_ids"] if re.search(r"(?<![\w:])" + re.escape(e) + r"(?![\w:])", txt) and e != task["evidence"]]
    return {"scope": keys, "consumes": cons}

def call_claude(model, prompt):
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", method="POST",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
        data=json.dumps({"model": model, "max_tokens": 2000, "temperature": 0,
                         "messages": [{"role": "user", "content": prompt}]}).encode())
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    return "".join(b.get("text", "") for b in out["content"]), out.get("usage", {})

def call_openai(model, prompt):
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    req = urllib.request.Request(base + "/chat/completions", method="POST",
        headers={"content-type": "application/json", "authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "none")},
        data=json.dumps({"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode())
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"], out.get("usage", {})

def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:
        j = {}
    return {"scope": [str(x) for x in j.get("scope", [])], "consumes": [str(x) for x in j.get("consumes", [])]}

def run(tasks_path, proposer, out_path):
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path)}
    with open(out_path, "a") as f:
        for line in open(tasks_path):
            t = json.loads(line)
            if t["id"] in done:
                continue
            usage, raw = {}, None
            if proposer == "parser":
                p = parser(t)
            elif proposer == "parser+templates":
                p = parser(t); p["scope"] = sorted(set(p["scope"]) | set(t["template"]))
                if t["type"] == "rta": p["consumes"] = sorted(set(p["consumes"]) | {"INT"})
            else:
                backend, _, model = proposer.partition(":")
                prompt = PROMPT.format(eid=t["evidence"], report=t["report"],
                                       cat="\n".join(f"{k}: {v}" for k, v in t["catalogue"].items()),
                                       evs=", ".join(e for e in t["evidence_ids"] if e != t["evidence"]))
                for attempt in range(5):
                    try:
                        raw, usage = (call_claude if backend == "claude" else call_openai)(model, prompt)
                        break
                    except Exception as ex:
                        time.sleep(2 ** attempt); raw = f"ERROR {ex}"
                p = parse_json(raw or "")
            f.write(json.dumps({"id": t["id"], "proposer": proposer, **p, "usage": usage, "raw": raw}) + "\n"); f.flush()

# ----------------------------------------------------------------------------- scoring
def score(tasks_path, prop_path):
    tasks = {json.loads(l)["id"]: json.loads(l) for l in open(tasks_path)}
    by = {}
    for line in open(prop_path):
        p = json.loads(line); t = tasks[p["id"]]
        true, prop = set(t["truth_scope"]), set(p["scope"])
        tc, pc = set(t["truth_consumes"]), set(p["consumes"])
        checked_ok = set(t["manifest"]) <= prop and set(t["template"]) <= prop and \
                     (t["type"] != "rta" or "INT" in pc)
        for key in (t["type"], f"style:{t['style']}", "ALL"):
            b = by.setdefault(key, dict(n=0, true=0, omitted=0, invented=0, items_with_omission=0,
                                        consume_missed=0, rejected_by_checks=0, omission_after_checks=0))
            b["n"] += 1; b["true"] += len(true); b["omitted"] += len(true - prop); b["invented"] += len(prop - true)
            b["items_with_omission"] += bool(true - prop) or bool(tc - pc)
            b["consume_missed"] += len(tc - pc)
            b["rejected_by_checks"] += not checked_ok
            b["omission_after_checks"] += checked_ok and (bool(true - prop) or bool(tc - pc))
    for b in by.values():
        b["omission_rate"] = round(b["omitted"] / b["true"], 4) if b["true"] else None
        b["items_with_omission_rate"] = round(b["items_with_omission"] / b["n"], 4)
        b["escalation_rate"] = round(b["rejected_by_checks"] / b["n"], 4)
        b["undetected_omission_rate"] = round(b["omission_after_checks"] / b["n"], 4)
    return by

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        gen(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "score":
        json.dump(score(sys.argv[2], sys.argv[3]), sys.stdout, indent=1)
