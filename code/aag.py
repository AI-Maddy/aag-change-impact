#!/usr/bin/env python3
"""Reference implementation of the deterministic core of the Agentic Assurance Graph (AAG).

Implements, over the hand-constructed brake-by-wire case study (case_study.json):
  Eq. (1)  evidence validity from content-addressed scopes;
  Eq. (2)  claim support, in the submitted form (vacuous for empty claims) and in the
           revised fail-closed form (non-empty, required evidence types, approval bound
           to a support snapshot and revoked on any change);
  Eq. (3)  invalidation closed transitively over evidence-to-evidence dependencies
           (outputs of evidence items are themselves scoped artifacts out(E));
  the deterministic scope checks applied to an A1 proposal (manifest containment and
  type-level scope templates); the dependency order that a re-verification plan must respect;
  and two baselines: code-centric trace links and a flat input-manifest approach.

No language model is involved anywhere in this file. Run: python3 aag.py [case_study.json]
"""
import hashlib, json, sys
from collections import OrderedDict

def h(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]

class AAG:
    def __init__(self, cs):
        self.cs = cs
        self.ev = cs["evidence"]
        self.claims = cs["claims"]
        # artifact contents are symbolic; version 0 before the change
        self.version = {a: 0 for a in cs["artifacts"]}
        self.pinned = {}          # e -> {artifact: hash at production time}
        self.approvals = {}       # claim -> support snapshot hash at approval time
        for e in self.ev:
            # derived inputs out(E') are not hashed here; Eq. (3) checks them through valid(E')
            self.pinned[e] = {a: self.hash_art(a) for a in self.scope(e, flat=True)}

    # ---- artifacts and scopes -------------------------------------------------
    def hash_art(self, a):
        if a.startswith("out:"):
            return h(a + ":" + self.result_hash(a[4:]))
        return h(f"{a}@{self.version[a]}")

    def result_hash(self, e):
        # an evidence result is stable while the evidence is valid; once invalid its
        # output is pending re-run and must be treated as changed
        return "r0" if self.valid_direct(e) else "pending"

    def scope(self, e, flat=False):
        d = self.ev[e]
        s = list(d["manifest"]) + list(d["extra"])
        if not flat:
            s += ["out:" + x for x in d.get("consumes", [])]
        return s

    def apply_change(self, change_set):
        for a in change_set:
            self.version[a] += 1

    # ---- Eq. (1): validity on the item's own (non-derived) inputs ---------------
    def valid_direct(self, e):
        return all(h(f"{a}@{self.version[a]}") == self.pinned[e][a]
                   for a in self.scope(e, flat=True))

    # ---- Eq. (3): validity closed over evidence-to-evidence dependencies --------
    def valid(self, e, _seen=None):
        _seen = set() if _seen is None else _seen
        if e in _seen:
            raise ValueError("cyclic evidence dependency at " + e)
        if not self.valid_direct(e):
            return False
        return all(self.valid(x, _seen | {e}) for x in self.ev[e].get("consumes", []))

    def invalidated(self, closed=True):
        f = self.valid if closed else self.valid_direct
        return [e for e in self.ev if not f(e)]

    # ---- Eq. (2): claim support -------------------------------------------------
    def sup_submitted(self, c, valid):
        """As submitted: AND over Ev(c) and Sub(c); vacuously true when both are empty."""
        d = self.claims[c]
        return all(valid(e) for e in d["ev"]) and all(self.sup_submitted(s, valid) for s in d["sub"])

    def snapshot(self, c):
        d = self.claims[c]
        parts = [e + ":" + ",".join(self.pinned[e][a] + ("" if self.valid(e) else "!")
                                    for a in sorted(self.pinned[e])) for e in d["ev"]]
        parts += ["sub:" + s + ":" + self.snapshot(s) for s in d["sub"]]
        return h("|".join(parts))

    def approve_all(self):
        for c in self.claims:
            if self.sup_revised(c, need_approval=False):
                self.approvals[c] = self.snapshot(c)

    def sup_revised(self, c, need_approval=True):
        """Revised, fail closed: non-empty support, every required evidence type present among
        valid direct evidence, all evidence valid (closed validity), all sub-claims supported,
        and a human approval recorded against the current support snapshot."""
        d = self.claims[c]
        if not d["ev"] and not d["sub"]:
            return False
        if not all(self.valid(e) for e in d["ev"]):
            return False
        types = {self.ev[e]["type"] for e in d["ev"] if self.valid(e)}
        if not set(d.get("req", [])) <= types:
            return False
        if not all(self.sup_revised(s, need_approval) for s in d["sub"]):
            return False
        if need_approval and self.approvals.get(c) != self.snapshot(c):
            return False
        return True

    # ---- deterministic checks on an A1 scope proposal ---------------------------
    def check_scope_proposal(self, e, proposed, links_to_claims=True):
        d = self.ev[e]
        man = set(d["manifest"])
        prop = set(proposed)
        missing_manifest = sorted(man - prop)
        tmpl = []
        for t in self.cs["scope_templates"].get(d["type"], []):
            if t.startswith("out:"):
                want = t[4:]
                ok = any(p.startswith("out:") and self.ev[p[4:]]["type"] == want for p in prop)
                if not ok:
                    tmpl.append(t)
            elif t not in prop:
                tmpl.append(t)
        extras = sorted(prop - man)
        if missing_manifest or tmpl:
            verdict = "reject"
        elif not extras and not links_to_claims:
            verdict = "auto-commit"
        else:
            verdict = "human-commit"
        return {"verdict": verdict, "missing_manifest": missing_manifest,
                "missing_template": tmpl, "beyond_manifest": extras}

    # ---- dependency order for re-verification -----------------------------------
    def reverification_layers(self):
        inv = set(self.invalidated(closed=True))
        layers, done = [], set()
        while inv - done:
            layer = sorted([e for e in inv - done
                            if all(x in done or x not in inv for x in self.ev[e].get("consumes", []))],
                           key=lambda x: int(x[1:]))
            if not layer:
                raise ValueError("cycle")
            layers.append(layer)
            done |= set(layer)
        return layers

def ancestors(claims, evset):
    hit = set()
    changed = True
    while changed:
        changed = False
        for c, d in claims.items():
            if c in hit:
                continue
            if set(d["ev"]) & evset or set(d["sub"]) & hit:
                hit.add(c); changed = True
    return sorted(hit, key=lambda x: int(x[1:]))

def main(path):
    cs = json.load(open(path))
    C = cs["change_set"]
    out = OrderedDict()

    g = AAG(cs)
    g.approve_all()
    assert all(g.sup_revised(c) for c in g.claims), "all claims supported before the change"
    out["before_change_all_supported"] = True
    g.apply_change(C)

    key = lambda x: int(x[1:])
    direct = sorted(g.invalidated(closed=False), key=key)
    closed = sorted(g.invalidated(closed=True), key=key)
    trace = sorted([e for e, arts in cs["trace_links"].items() if not e.startswith("_") and set(arts) & set(C)], key=key)
    manifest = sorted([e for e, d in cs["evidence"].items() if set(d["manifest"]) & set(C)], key=key)

    out["change_set"] = C
    out["invalidated"] = OrderedDict([
        ("trace_link_baseline", trace),
        ("input_manifest_baseline", manifest),
        ("aag_direct_scopes_as_submitted", direct),
        ("aag_closed_revised", closed),
    ])
    out["affected_claims"] = OrderedDict((k, ancestors(cs["claims"], set(v))) for k, v in out["invalidated"].items())
    out["unsupported_revised_eq2"] = [c for c in sorted(g.claims, key=key) if not g.sup_revised(c)]
    out["approvals_revoked"] = [c for c in sorted(g.claims, key=key) if g.approvals.get(c) != g.snapshot(c)]
    out["reverification_layers"] = g.reverification_layers()

    # empty-claim check (Reviewer 3)
    g2 = AAG(json.loads(json.dumps(cs)))
    g2.claims["Gx"] = {"desc": "claim with no evidence", "ev": [], "sub": [], "req": []}
    g2.approvals["Gx"] = g2.snapshot("Gx")
    out["empty_claim"] = {"submitted_eq2": g2.sup_submitted("Gx", g2.valid_direct),
                          "revised_eq2": g2.sup_revised("Gx")}

    # scope proposals A1 could make (Reviewers 2 and 3)
    g3 = AAG(cs)
    props = OrderedDict()
    props["E4 = manifest only (tool scope)"] = g3.check_scope_proposal("E4", cs["evidence"]["E4"]["manifest"])
    props["E4 = manifest + out(E5)"] = g3.check_scope_proposal("E4", cs["evidence"]["E4"]["manifest"] + ["out:E5"])
    props["E5 = manifest only (image)"] = g3.check_scope_proposal("E5", cs["evidence"]["E5"]["manifest"])
    props["E5 = full declared scope"] = g3.check_scope_proposal("E5", g3.scope("E5"))
    props["E6 drops one partition entry"] = g3.check_scope_proposal("E6", ["mpu.part.BrakeCtl", "mpu.part.WSpeed"])
    props["E9 without security view"] = g3.check_scope_proposal("E9", ["can.matrix"])
    props["E9 full declared scope"] = g3.check_scope_proposal("E9", g3.scope("E9"))
    props["E11 = manifest, no claim link"] = g3.check_scope_proposal("E11", ["code.BrakeCtl"], links_to_claims=False)
    out["a1_scope_checks"] = props

    json.dump(out, sys.stdout, indent=2); print()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "case_study.json")
