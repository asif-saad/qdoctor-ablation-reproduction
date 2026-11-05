"""Cross-check the numpy simulation pipeline against Qiskit (Statevector / Operator) and Qiskit Aer.

Checks:
 1. DEP / DEPcode numpy input states == Statevector of the equivalent Qiskit preparation circuit.
 2. Mutant unitaries from prefix/suffix products == Operator of the rebuilt mutant circuit.
 3. Exact numpy output distributions (input -> mutant -> basis change) == Aer shot distributions
    (JSD between numpy-sampled and Aer-sampled counts is at the shot-noise level).
Output: results/verify_sim.txt
"""
from __future__ import annotations

import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuits import ops_to_circuit  # noqa: E402
import probes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def dep_circuit_paper(n, rng):
    """Re-draw the same random numbers as probes.dep_inputs and build the Qiskit circuit."""
    th = rng.uniform(0, math.pi, n)
    ph = rng.uniform(-math.pi, math.pi, n)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(th[q], q)
        qc.rz(ph[q], q)
    hub = int(rng.integers(n))
    g = ["cx", "cy", "cz", "ch"][int(rng.integers(4))]
    for t in range(n):
        if t != hub:
            getattr(qc, g)(hub, t)
    return qc


def dep_circuit_code(n, rng, depth=2):
    th = rng.uniform(0, math.pi / 2, n)
    ph = rng.uniform(-math.pi, math.pi, n)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(th[q], q)
        qc.rz(ph[q], q)
    anchor = int(rng.integers(n))
    targets = [q for q in range(n) if q != anchor]
    order = list(rng.permutation(targets))
    for _ in range(depth):
        if not order:
            order = list(rng.permutation(targets))
        t = int(order.pop(0))
        g = ["rx", "ry", "rz", "h"][int(rng.integers(4))]
        if g == "h":
            qc.ch(anchor, t)
        else:
            getattr(qc, "c" + g)(rng.uniform(0, math.pi), anchor, t)
    return qc


def main() -> None:
    lines = []
    log = lambda s: (print(s), lines.append(s))  # noqa: E731

    # 1. input states
    worst = 0.0
    for n in (3, 4):
        for seed in range(50):
            for gen, ref in (("dep", dep_circuit_paper), ("depcode", dep_circuit_code)):
                states, _ = probes.INPUTS[gen](n, np.random.default_rng(seed), k=1)
                qc = ref(n, np.random.default_rng(seed))
                sv = Statevector(qc).data
                worst = max(worst, abs(abs(np.vdot(sv, states[0])) - 1))
    log(f"[1] input states: max |1-|<qiskit|numpy>|| over 200 probes = {worst:.2e}")

    # 2. mutant unitaries
    with open(ROOT / "data" / "mutants.pkl", "rb") as f:
        bundle = pickle.load(f)
    muts = random.Random(0).sample(bundle["mutants"], 200)
    worst = max(np.abs(Operator(ops_to_circuit(m["ops"], m["n"])).data - m["U"]).max() for m in muts)
    log(f"[2] mutant unitaries: max |Operator - prefix/suffix| over 200 mutants = {worst:.2e}")

    # 3. Aer shot distributions vs numpy
    sim = AerSimulator(method="statevector", seed_simulator=7)
    rng = np.random.default_rng(1)
    js_np_aer, js_np_np = [], []
    for m in muts[:40]:
        n = m["n"]
        for gen in ("basis", "dep"):
            states, meta = probes.INPUTS[gen](n, rng, k=1)
            meas = [rng.choice(["Z", "X", "Y"])]
            p = probes.output_probs(m["U"], states, meas, n)[0]
            qc = QuantumCircuit(n)
            qc.initialize(states[0], range(n))
            qc.compose(ops_to_circuit(m["ops"], n), inplace=True)
            for q in range(n):
                if meas[0] == "Y":
                    qc.sdg(q)
                if meas[0] in ("X", "Y"):
                    qc.h(q)
            qc.measure_all()
            counts = sim.run(transpile(qc, sim), shots=probes.SHOTS).result().get_counts()
            aer = np.zeros(2 ** n)
            for bits, c in counts.items():
                aer[int(bits, 2)] = c / probes.SHOTS
            a = probes.sample_dists(p[None], rng)[0]
            b = probes.sample_dists(p[None], rng)[0]
            js_np_aer.append(float(probes.jsd(a, aer)))
            js_np_np.append(float(probes.jsd(a, b)))
    log(f"[3] JSD(numpy-sampled, Aer-sampled): mean={np.mean(js_np_aer):.2e} max={np.max(js_np_aer):.2e}")
    log(f"    JSD(numpy-sampled, numpy-sampled): mean={np.mean(js_np_np):.2e} max={np.max(js_np_np):.2e}")
    log("    (the two should match: Aer differs from numpy only by shot noise; both far below tau=2.5e-3)")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "verify_sim.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
