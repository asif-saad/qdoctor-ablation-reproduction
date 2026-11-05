"""Probe components: input-state generators, measurement pools, shot sampling and the JSD oracle.

All states are numpy vectors in Qiskit's little-endian order (qubit q <-> bit q of the index).
"""
from __future__ import annotations

import math

import numpy as np

SHOTS = 20_000
NUM_PROBES = 20

# ---------------------------------------------------------------- 1-qubit matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
SDG = np.array([[1, 0], [0, -1j]], dtype=complex)


def rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)


def apply_controlled(psi: np.ndarray, U: np.ndarray, c: int, t: int, n: int) -> np.ndarray:
    """Apply controlled-U (control c, target t) to one n-qubit state vector."""
    out = psi.reshape([2] * n).copy()
    ac, at = n - 1 - c, n - 1 - t          # tensor axis of qubit q is n-1-q
    sl = [slice(None)] * n
    sl[ac] = 1
    sub = out[tuple(sl)]                    # control = 1 block (axis ac removed)
    at2 = at if at < ac else at - 1
    sub = np.moveaxis(np.tensordot(U, sub, axes=([1], [at2])), 0, at2)
    out[tuple(sl)] = sub
    return out.reshape(-1)


def product_state(local: list[np.ndarray]) -> np.ndarray:
    """|v_{n-1}> ⊗ ... ⊗ |v_0>  (local[q] is qubit q's state)."""
    psi = np.array([1.0 + 0j])
    for v in local:
        psi = np.kron(v, psi)
    return psi


# ---------------------------------------------------------------- input generators
def basis_inputs(n: int, rng: np.random.Generator, k: int = NUM_PROBES):
    """Quito/QuSBT-style baseline (released QDoctor code): computational-basis states sampled
    without replacement; if k >= 2^n every basis state is used once (shuffled)."""
    N = 2 ** n
    idx = rng.permutation(N) if k >= N else rng.choice(N, size=k, replace=False)
    states = np.zeros((len(idx), N), dtype=complex)
    states[np.arange(len(idx)), idx] = 1.0
    return states, [{"basis": int(i)} for i in idx]


_DEP_GATES = {"cx": X, "cy": Y, "cz": Z, "ch": H}


def dep_inputs(n: int, rng: np.random.Generator, k: int = NUM_PROBES):
    """Difference-Enhancing Probe input, as described in the paper (§3.2, Fig. 3):
    Ry(θ_i) Rz(φ_i) on every qubit, then one gate type from {CX,CY,CZ,CH} applied from a random
    hub qubit t to every other qubit (star topology)."""
    states, meta = [], []
    for _ in range(k):
        th = rng.uniform(0, math.pi, n)
        ph = rng.uniform(-math.pi, math.pi, n)
        psi = product_state([rz(ph[q]) @ ry(th[q]) @ np.array([1, 0], dtype=complex) for q in range(n)])
        hub = int(rng.integers(n))
        g = ["cx", "cy", "cz", "ch"][int(rng.integers(4))]
        for tgt in range(n):
            if tgt != hub:
                psi = apply_controlled(psi, _DEP_GATES[g], hub, tgt, n)
        states.append(psi)
        meta.append({"hub": hub, "gate": g})
    return np.array(states), meta


def depcode_inputs(n: int, rng: np.random.Generator, k: int = NUM_PROBES, depth: int = 2):
    """Sensitivity variant mirroring the authors' released ConcentratedEntangledInputMutator with
    their example config (mode1, controlled_gate='mix', entangle_depth=2, theta_zero_frac=0):
    Ry(θ), θ~U[0,π/2]; Rz(φ), φ~U[-π,π]; `depth` controlled gates from one anchor, each gate drawn
    from {CRX,CRY,CRZ,CH} with rotation angle ~U[0,π]."""
    states, meta = [], []
    for _ in range(k):
        th = rng.uniform(0, math.pi / 2, n)
        ph = rng.uniform(-math.pi, math.pi, n)
        psi = product_state([rz(ph[q]) @ ry(th[q]) @ np.array([1, 0], dtype=complex) for q in range(n)])
        anchor = int(rng.integers(n))
        targets = [q for q in range(n) if q != anchor]
        order = list(rng.permutation(targets))
        for layer in range(depth):
            if not order:
                order = list(rng.permutation(targets))
            tgt = int(order.pop(0))
            g = ["rx", "ry", "rz", "h"][int(rng.integers(4))]
            U = H if g == "h" else {"rx": rx, "ry": ry, "rz": rz}[g](rng.uniform(0, math.pi))
            psi = apply_controlled(psi, U, anchor, tgt, n)
        states.append(psi)
        meta.append({"anchor": anchor})
    return np.array(states), meta


INPUTS = {"basis": basis_inputs, "dep": dep_inputs, "depcode": depcode_inputs}


# ---------------------------------------------------------------- measurement pools
def _tensor_power(m: np.ndarray, n: int) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for _ in range(n):
        out = np.kron(m, out)
    return out


def basis_change(name: str, n: int) -> np.ndarray:
    """Pre-measurement rotation. X: H on all qubits; Y: Sdg then H on all qubits;
    H: identical to X, exactly as `measure_h` in the released QDoctor code."""
    if name == "Z":
        return np.eye(2 ** n, dtype=complex)
    if name in ("X", "H"):
        return _tensor_power(H, n)
    if name == "Y":
        return _tensor_power(H @ SDG, n)
    raise ValueError(name)


def measurement_choices(pool: str, k: int, rng: np.random.Generator) -> list[str]:
    if pool == "Z":
        return ["Z"] * k
    if pool == "XYH":                       # random_balanced: concatenated shuffled bags
        out = []
        while len(out) < k:
            out.extend(rng.permutation(["X", "Y", "H"]).tolist())
        return out[:k]
    raise ValueError(pool)


# ---------------------------------------------------------------- execution + oracle
def output_probs(U: np.ndarray, states: np.ndarray, meas: list[str], n: int) -> np.ndarray:
    out = states @ U.T                                   # each row: U |psi>
    rot = {m: basis_change(m, n) for m in set(meas)}
    out = np.stack([rot[m] @ out[i] for i, m in enumerate(meas)])
    p = np.abs(out) ** 2
    return p / p.sum(axis=1, keepdims=True)


def sample_dists(p: np.ndarray, rng: np.random.Generator, shots: int = SHOTS) -> np.ndarray:
    return rng.multinomial(shots, p) / shots


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise base-2 Jensen–Shannon divergence, eps-clipped as in QDoctor's metrics.py."""
    p = np.clip(p, eps, 1.0); p = p / p.sum(axis=-1, keepdims=True)
    q = np.clip(q, eps, 1.0); q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log2(p / m), axis=-1) + 0.5 * np.sum(q * np.log2(q / m), axis=-1)
