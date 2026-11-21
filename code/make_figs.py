#!/usr/bin/env python3
"""Figures and LaTeX tables from bench results. Usage: python3 make_figs.py results_dir out_dir"""
import json, sys, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "bench_500_seed*.json")))]
M = ["trace", "manifest", "aag_direct", "aag_closed", "aag_cutoff"]
LBL = {"trace": "Trace links", "manifest": "Input manifest", "aag_direct": "AAG, direct scopes",
       "aag_closed": "AAG, closed", "aag_cutoff": "AAG, closed + cut-off", "aag_closed_file": "AAG, closed, file-level"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})

def pooled(variant, bucket, m, key):
    vals = [r["methods"][variant][bucket][m][key] for r in runs]
    return sum(vals) / len(vals), min(vals), max(vals)

def tot(variant, bucket, key):
    return sum(r["methods"][variant][bucket][key] for r in runs)

# ---- Table: overall results, both interference variants
lines = []
for variant in ("measured", "analytic"):
    need = tot(variant, "ALL", "needed_hours")
    for m in M + ["aag_closed_file"]:
        rec = pooled(variant, "ALL", m, "recall")
        pre = pooled(variant, "ALL", m, "precision")
        cm = sum(r["methods"][variant]["ALL"][m]["scenarios_with_critical_miss"] for r in runs)
        fs = sum(r["methods"][variant]["ALL"][m]["scenarios_with_false_support"] for r in runs)
        hrs = sum(r["methods"][variant]["ALL"][m]["reverification_hours"] for r in runs)
        lines.append(f"{variant} & {LBL[m]} & {rec[0]:.3f} & {pre[0]:.3f} & {cm} & {fs} & {hrs/need:.2f} \\\\")
n_scen = tot("measured", "ALL", "scenarios")
with open(os.path.join(OUT, "tab_overall.tex"), "w") as f:
    f.write(f"%% scenarios per variant: {n_scen}\n")
    f.write("\n".join(lines) + "\n")

# ---- Table: recall per change type (measured variant)
rows = []
changes = [c for c in runs[0]["methods"]["measured"] if c != "ALL"]
for c in changes:
    st = tot("measured", c, "stale_items")
    rec = [pooled("measured", c, m, "recall")[0] for m in M]
    rows.append(f"{c.replace('_', ' ')} & {st} & " + " & ".join("--" if r is None else f"{r:.2f}" for r in rec) + " \\\\")
with open(os.path.join(OUT, "tab_per_change.tex"), "w") as f:
    f.write("\n".join(rows) + "\n")

# ---- Figure: recall vs relative re-verification effort
fig, ax = plt.subplots(figsize=(3.4, 2.3))
mk = {"measured": "o", "analytic": "s"}
for variant in ("measured", "analytic"):
    need = tot(variant, "ALL", "needed_hours")
    for m in M + ["aag_closed_file"]:
        rec = pooled(variant, "ALL", m, "recall")[0]
        hrs = sum(r["methods"][variant]["ALL"][m]["reverification_hours"] for r in runs) / need
        ax.scatter(hrs, rec, marker=mk[variant], s=22, facecolors="none" if variant == "analytic" else "k", edgecolors="k")
        if variant == "measured":
            off = {"aag_cutoff": (-75, -18), "aag_closed": (-40, -32), "aag_closed_file": (-20, -18),
                   "aag_direct": (-60, 8), "manifest": (-20, 8), "trace": (-10, 8)}[m]
            ax.annotate(LBL[m], (hrs, rec), fontsize=6.5, xytext=off, textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4") if m.startswith("aag_c") else None)
ax.set_xlabel("Re-verification effort / effort of truly stale items")
ax.set_ylabel("Recall of stale evidence")
ax.set_ylim(0, 1.05); ax.set_xlim(0.5, 3.4); ax.grid(alpha=0.3)
ax.scatter([], [], marker="o", c="k", label="measured interference")
ax.scatter([], [], marker="s", facecolors="none", edgecolors="k", label="analytic interference")
ax.legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_recall_cost.pdf")); plt.close(fig)

# ---- Figure: scope-error sensitivity
fig, ax1 = plt.subplots(figsize=(3.4, 2.2))
ps = [0.0, 0.05, 0.1, 0.2, 0.3]
for checks, ls in ((False, "--"), (True, "-")):
    rec = [sum(r["scope_errors"][f"p={p}|checks={checks}"]["recall"] for r in runs) / len(runs) for p in ps]
    ax1.plot(ps, rec, "k" + ls, marker="o", ms=3, label=f"A1 scopes, {'with' if checks else 'without'} checks")
ax1.set_xlabel("Probability that A1 omits a true scope entry")
ax1.set_ylabel("Recall of stale evidence")
ax2 = ax1.twinx()
esc = [sum(r["scope_errors"][f"p={p}|checks=True"]["escalation_rate"] for r in runs) / len(runs) for p in ps]
ax2.plot(ps, esc, color="0.5", ls=":", marker="^", ms=3, label="proposals escalated (right)")
ax2.set_ylabel("Escalated to human", color="0.4")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="lower left")
ax1.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scope_err.pdf")); plt.close(fig)

# ---- scope error table numbers
with open(os.path.join(OUT, "scope_err.txt"), "w") as f:
    for p in ps:
        for checks in (False, True):
            k = f"p={p}|checks={checks}"
            rec = sum(r["scope_errors"][k]["recall"] for r in runs) / len(runs)
            cm = sum(r["scope_errors"][k]["missed_critical"] for r in runs)
            sm = sum(r["scope_errors"][k]["scenarios_with_miss"] for r in runs)
            esc = sum(r["scope_errors"][k]["escalation_rate"] for r in runs) / len(runs)
            f.write(f"{k}: recall={rec:.4f} critical_missed={cm} scen_with_miss={sm} escalation={esc:.3f}\n")
print("written to", OUT)

# ---- Table: recall per evidence type (measured variant, pooled)
names = {"rta": "Response-time analysis", "interference": "Interference analysis", "hil": "HIL timing test",
         "mpu": "MPU partition check", "tara": "TARA", "unit": "Unit tests", "static": "Static analysis",
         "e2e": "E2E protection tests", "review": "Hazard analysis review"}
pt = [r["methods"]["measured"]["ALL"]["per_type"] for r in runs]
rows = []
for ty in sorted(pt[0], key=lambda t: -sum(p[t]["stale"] for p in pt)):
    st = sum(p[ty]["stale"] for p in pt)
    rec = [sum(p[ty][m] * p[ty]["stale"] for p in pt) / st for m in M]
    rows.append(f"{names.get(ty, ty)} & {st} & " + " & ".join(f"{x:.2f}" for x in rec) + " \\\\")
with open(os.path.join(OUT, "tab_per_type.tex"), "w") as f:
    f.write("\n".join(rows) + "\n")

# ---- Table: scalability
sc = json.load(open(os.path.join(RES, "scale.json")))
rows = [f"{x['evidence']:,} & {x['claims']:,} & {x['changed']} & {x['invalidated_median']:,} & {x['median_s']*1000:.1f} \\\\" for x in sc]
with open(os.path.join(OUT, "tab_scale.tex"), "w") as f:
    f.write("\n".join(rows) + "\n")
