"""Independent audit (optional, ~2 min; not part of run_all.sh).

1. Re-labels all 169 equivalent + 600 random inequivalent mutants with qiskit Operator.equiv.
2. Re-runs 15 mutants that A2 (DEP + Z) misses in run 0 through Qiskit Aer with fresh DEP probes,
   using scipy jensenshannon instead of our jsd(), and checks they are missed under Z and killed under X.
Output: results/audit_independent.txt
"""
import pickle, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from circuits import ops_to_circuit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator
from scipy.spatial.distance import jensenshannon

d = pickle.load(open("data/mutants.pkl", "rb"))
orig, muts = d["originals"], d["mutants"]
rng = np.random.default_rng(7)
eq = [m for m in muts if m["equivalent"]]
ineq = [m for m in muts if not m["equivalent"]]
sample = eq + [ineq[i] for i in rng.choice(len(ineq), 600, replace=False)]
bad = 0
for m in sample:
    o = orig[m["program"]]
    U0 = Operator(ops_to_circuit(o["ops"], m["n"]))
    U1 = Operator(ops_to_circuit(m["ops"], m["n"]))
    if U0.equiv(U1, atol=1e-10) != m["equivalent"]:
        bad += 1
print(f"label check: {len(sample)} mutants ({len(eq)} equiv), mismatches vs qiskit Operator.equiv = {bad}")

# Aer check of kill verdicts: Z-invisible-type misses under A2 and kills under A4
mj = pd.read_csv("results/raw/maxjsd.csv")
r0 = mj[mj.run == 0].pivot(index="mid", columns="config", values="max_jsd")
byid = {m["mid"]: m for m in muts}
missA2 = [i for i in r0.index if not byid[i]["equivalent"] and r0.loc[i, "A2"] <= 0.0025][:15]
sim = AerSimulator()
def aer_dist(ops, n, prep, meas, shots=20000, seed=0):
    qc = QuantumCircuit(n); qc.compose(prep, inplace=True)
    qc.compose(ops_to_circuit(ops, n), inplace=True); qc.compose(meas, inplace=True); qc.measure_all()
    c = sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed).result().get_counts()
    p = np.zeros(2**n)
    for k, v in c.items(): p[int(k.replace(" ", ""), 2)] = v
    return p / shots
def jsd(p, q): return jensenshannon(p, q, base=2) ** 2
agree = 0; tot = 0
for i in missA2:
    m = byid[i]; n = m["n"]; o = orig[m["program"]]
    best_z = best_x = 0
    for t in range(20):
        r = np.random.default_rng([99, i, t]); prep = QuantumCircuit(n)
        for q in range(n): prep.ry(r.uniform(0, np.pi), q); prep.rz(r.uniform(-np.pi, np.pi), q)
        hub = r.integers(n)
        for q in range(n):
            if q != hub: prep.cx(hub, q)
        z = QuantumCircuit(n); x = QuantumCircuit(n); x.h(range(n))
        best_z = max(best_z, jsd(aer_dist(o["ops"], n, prep, z, seed=t), aer_dist(m["ops"], n, prep, z, seed=t + 100)))
        best_x = max(best_x, jsd(aer_dist(o["ops"], n, prep, x, seed=t), aer_dist(m["ops"], n, prep, x, seed=t + 100)))
    tot += 1; agree += (best_z <= 0.0025) and (best_x > 0.0025)
    print(f"mid {i} {m['program']} {m['desc']}: Aer DEP+Z max JSD {best_z:.2e} (miss)  DEP+X max JSD {best_x:.2e}")
print(f"Aer: {agree}/{tot} A2-missed mutants are also missed by DEP+Z and killed by DEP+X on Aer")
