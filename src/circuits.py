"""Build the 8 original benchmark circuits (GHZ, QFT, Grover, W-state at 3 and 4 qubits).

Circuits come from MQT Bench (target-independent level), measurements and barriers are
removed, and each circuit is transpiled to the elementary gate set used by the mutation
operators so that composite gates (Grover's Q) become individually mutable gates.

Output: data/originals.pkl  -> {name: [(gate, qubits, params), ...]}
        data/originals/<name>.qasm  (human-readable copy)
"""
from __future__ import annotations

import pickle
from pathlib import Path

from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit, qasm2, transpile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ALGORITHMS = ["ghz", "qft", "grover", "wstate"]
SIZES = [3, 4]

# Elementary gate set shared with the mutation operators (src/mutants.py).
BASIS = ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx",
         "p", "rx", "ry", "rz",
         "cx", "cy", "cz", "ch", "swap",
         "cp", "crx", "cry", "crz",
         "ccx", "cswap"]

Op = tuple  # (name: str, qubits: tuple[int, ...], params: tuple[float, ...])


def circuit_to_ops(qc: QuantumCircuit) -> list[Op]:
    ops = []
    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        params = tuple(float(p) for p in inst.operation.params)
        ops.append((name, qubits, params))
    return ops


def ops_to_circuit(ops: list[Op], n: int) -> QuantumCircuit:
    from qiskit.circuit.library import get_standard_gate_name_mapping
    lib = get_standard_gate_name_mapping()
    qc = QuantumCircuit(n)
    for name, qubits, params in ops:
        gate = lib[name]
        if params:
            gate = type(gate)(*params)
        qc.append(gate, list(qubits))
    return qc


def build_originals() -> dict[str, dict]:
    out = {}
    for alg in ALGORITHMS:
        for n in SIZES:
            qc = get_benchmark(alg, BenchmarkLevel.INDEP, n)
            qc = qc.remove_final_measurements(inplace=False)
            qc.data = [d for d in qc.data if d.operation.name != "barrier"]
            t = transpile(qc, basis_gates=BASIS, optimization_level=0, seed_transpiler=0)
            name = f"{alg}{n}"
            out[name] = {"algorithm": alg, "n": n, "ops": circuit_to_ops(t)}
    return out


def main() -> None:
    DATA.mkdir(exist_ok=True)
    (DATA / "originals").mkdir(exist_ok=True)
    originals = build_originals()
    with open(DATA / "originals.pkl", "wb") as f:
        pickle.dump(originals, f)
    for name, c in originals.items():
        qc = ops_to_circuit(c["ops"], c["n"])
        (DATA / "originals" / f"{name}.qasm").write_text(qasm2.dumps(qc))
        counts = {}
        for g, _, _ in c["ops"]:
            counts[g] = counts.get(g, 0) + 1
        print(f"{name:8s} n={c['n']} gates={len(c['ops']):4d} {counts}")


if __name__ == "__main__":
    main()
