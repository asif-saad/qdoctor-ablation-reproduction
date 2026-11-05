"""Generate and label single-gate mutants (Muskit-style Add / Remove / Replace).

1. Enumerate the full single-mutation space of each original circuit.
2. Label every mutant equivalent / inequivalent by unitary comparison up to global phase
   (phase-invariant normalized Frobenius distance < EPS, as in QDoctor §4.3.3).
3. Keep a seeded uniform sample of at most CAP mutants per (circuit, operator).

Output: data/mutants.pkl            -> list of dicts (sampled mutants with unitaries)
        data/mutant_space_summary.csv -> counts over the FULL space
        data/mutants_summary.csv      -> counts over the sample
"""
from __future__ import annotations

import itertools
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qsim import embedded_matrix, phase_invariant_distance, prefix_products, suffix_products  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EPS = 1e-12
CAP = 1000         # max sampled mutants per (circuit, operator)
SEED = 2026
ADD_ANGLE = math.pi / 4

CATEGORIES = {
    "1q": ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx"],
    "1qp": ["p", "rx", "ry", "rz"],
    "2q": ["cx", "cy", "cz", "ch", "swap"],
    "2qp": ["cp", "crx", "cry", "crz"],
    "3q": ["ccx", "cswap"],
}
ARITY = {"1q": 1, "1qp": 1, "2q": 2, "2qp": 2, "3q": 3}
PARAMETRIC = {"1qp", "2qp"}
CAT_OF = {g: c for c, gs in CATEGORIES.items() for g in gs}


def enumerate_mutations(ops: list, n: int):
    """Yield (operator, position, description, new_gate_or_None)."""
    L = len(ops)
    for i in range(L):
        yield "remove", i, f"remove {ops[i][0]}{list(ops[i][1])}@{i}", None
    for i, (name, q, p) in enumerate(ops):
        for g in CATEGORIES[CAT_OF[name]]:
            if g != name:
                yield "replace", i, f"replace {name}->{g}{list(q)}@{i}", (g, q, p)
    for cat, gates in CATEGORIES.items():
        params = (ADD_ANGLE,) if cat in PARAMETRIC else ()
        for qs in itertools.permutations(range(n), ARITY[cat]):
            for g in gates:
                for pos in range(L + 1):
                    yield "add", pos, f"add {g}{list(qs)}@{pos}", (g, qs, params)


def mutant_ops(ops, operator, pos, new):
    if operator == "remove":
        return ops[:pos] + ops[pos + 1:]
    if operator == "replace":
        return ops[:pos] + [new] + ops[pos + 1:]
    return ops[:pos] + [new] + ops[pos:]


def label_space(name: str, circ: dict) -> list[dict]:
    ops, n = circ["ops"], circ["n"]
    P, S = prefix_products(ops, n), suffix_products(ops, n)
    U = P[-1]
    seen, rows = set(), []
    for operator, pos, desc, new in enumerate_mutations(ops, n):
        m_ops = mutant_ops(ops, operator, pos, new)
        key = tuple(m_ops)
        if key in seen:
            continue
        seen.add(key)
        if operator == "remove":
            V = S[pos + 1] @ P[pos]
        else:
            G = embedded_matrix(new[0], new[1], new[2], n)
            V = S[pos + 1] @ G @ P[pos] if operator == "replace" else S[pos] @ G @ P[pos]
        d = phase_invariant_distance(U, V)
        rows.append({"program": name, "algorithm": circ["algorithm"], "n": n,
                     "operator": operator, "desc": desc, "ops": m_ops,
                     "dist": d, "equivalent": d < EPS, "U": V})
    return rows


def main() -> None:
    with open(DATA / "originals.pkl", "rb") as f:
        originals = pickle.load(f)
    rng = random.Random(SEED)
    full_summary, sample = [], []
    for name, circ in originals.items():
        rows = label_space(name, circ)
        dists = np.array([r["dist"] for r in rows])
        near = int(((dists >= EPS) & (dists < 1e-6)).sum())
        print(f"{name:8s} space={len(rows):6d} eqv={int((dists < EPS).sum()):5d} "
              f"ambiguous(1e-12<=d<1e-6)={near} max_eqv_d={dists[dists < EPS].max() if (dists < EPS).any() else 0:.1e}")
        for op in ("add", "remove", "replace"):
            grp = [r for r in rows if r["operator"] == op]
            full_summary.append({"program": name, "operator": op, "total": len(grp),
                                 "equivalent": sum(r["equivalent"] for r in grp)})
            picked = grp if len(grp) <= CAP else rng.sample(grp, CAP)
            sample.extend(picked)
    for i, r in enumerate(sample):
        r["mid"] = i
    pd.DataFrame(full_summary).to_csv(DATA / "mutant_space_summary.csv", index=False)
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("ops", "U")} for r in sample])
    summ = df.groupby(["program", "operator"]).agg(total=("mid", "size"), equivalent=("equivalent", "sum")).reset_index()
    summ.to_csv(DATA / "mutants_summary.csv", index=False)
    df.drop(columns=[]).to_csv(DATA / "mutants_index.csv", index=False)
    with open(DATA / "mutants.pkl", "wb") as f:
        pickle.dump({"originals": originals, "mutants": sample}, f)
    print(summ.to_string())
    print(f"TOTAL sampled: {len(df)}  equivalent: {int(df.equivalent.sum())}  inequivalent: {int((~df.equivalent).sum())}")


if __name__ == "__main__":
    main()
