"""Small numpy helpers for exact simulation of op-list circuits (Qiskit little-endian order)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import Operator

_LIB = get_standard_gate_name_mapping()


def make_gate(name: str, params: tuple = ()):
    gate = _LIB[name]
    return type(gate)(*params) if params else gate


@lru_cache(maxsize=None)
def embedded_matrix(name: str, qubits: tuple, params: tuple, n: int) -> np.ndarray:
    """Full 2^n x 2^n matrix of one gate acting on `qubits` of an n-qubit register."""
    qc = QuantumCircuit(n)
    qc.append(make_gate(name, params), list(qubits))
    return Operator(qc).data


def prefix_products(ops: list, n: int) -> list[np.ndarray]:
    """P[i] = G_{i-1} ... G_0  (P[0] = I), length len(ops)+1."""
    P = [np.eye(2 ** n, dtype=complex)]
    for name, q, p in ops:
        P.append(embedded_matrix(name, q, p, n) @ P[-1])
    return P


def suffix_products(ops: list, n: int) -> list[np.ndarray]:
    """S[i] = G_{L-1} ... G_i  (S[L] = I), length len(ops)+1."""
    L = len(ops)
    S = [None] * (L + 1)
    S[L] = np.eye(2 ** n, dtype=complex)
    for i in range(L - 1, -1, -1):
        name, q, p = ops[i]
        S[i] = S[i + 1] @ embedded_matrix(name, q, p, n)
    return S


def unitary(ops: list, n: int) -> np.ndarray:
    return prefix_products(ops, n)[-1]


def phase_invariant_distance(U: np.ndarray, V: np.ndarray) -> float:
    """min_phi ||U - e^{i phi} V||_F / ||U||_F  with phi* = arg Tr(V^dag U)  (paper §4.3.3)."""
    tr = np.trace(V.conj().T @ U)
    phase = np.exp(1j * np.angle(tr)) if abs(tr) > 0 else 1.0
    return float(np.linalg.norm(U - phase * V) / np.linalg.norm(U))
