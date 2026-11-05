# Reproducing QDoctor's Input–Measurement Ablation

Code for the CMPS 7010 Research Seminar project (Tulane University) by Asif Mohammed Saad.
It reproduces the input × measurement ablation of QDoctor (Wang et al., ASE 2026) on
single-gate mutants of 3- and 4-qubit GHZ, QFT, Grover and W-state circuits from MQT Bench.

## Setup

Requires Python 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
./run_all.sh
```

The full pipeline takes under a minute on a multi-core laptop. All randomness is seeded,
so every run gives the same numbers.

| Step | Script | Output |
|------|--------|--------|
| 1 | `src/circuits.py` | original circuits (`data/originals.pkl`, `data/originals/*.qasm`) |
| 2 | `src/mutants.py` | mutant space, sampled mutants and equivalence labels (`data/`) |
| 3 | `src/verify_sim.py` | check of our simulator against Qiskit Aer (`results/verify_sim.txt`) |
| 4 | `src/run_ablation.py --runs 10` | probe results for A1–A4, S2, S4 over 10 runs (`results/raw/`) |
| 5 | `src/analyze.py` | tables (`results/*.csv`), figures (`results/fig_*.png`), summary `results/RESULTS.md` |

`src/qsim.py` (state-vector and unitary simulation) and `src/probes.py` (inputs, measurement
bases, JSD oracle) are shared modules used by the steps above.

### Optional: independent audit

```bash
python src/audit_independent.py
```

Run it from the repository root after `run_all.sh`; it takes about two minutes. It re-checks
the equivalence labels with Qiskit's `Operator.equiv`, and re-runs a sample of missed mutants
on Qiskit Aer with SciPy's Jensen–Shannon implementation in place of ours
(`results/audit_independent.txt`).
