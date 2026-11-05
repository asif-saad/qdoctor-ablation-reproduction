"""Run the input × measurement ablation over all sampled mutants for several independent runs.

Configs (inputs, measurement pool):
  A1 = (basis, Z)    A2 = (dep, Z)    A3 = (basis, XYH)    A4 = (dep, XYH)      <- paper RQ2
  S2 = (depcode, Z)  S4 = (depcode, XYH)                                         <- sensitivity
Within one (run, mutant) the configs that share an input family share the same input states.

For each (run, mutant, config) we store every probe's JSD between the sampled reference and the
sampled mutant distribution (20,000 shots each), plus the noise-free JSD. The kill decision for
any threshold τ is then  max_probe JSD > τ, so thresholds are applied later in analyze.py.

Output: results/raw/probe_jsd.npz  (arrays indexed [run, mutant, config, probe]; NaN = no probe)
        results/raw/maxjsd.csv      (long format: run, mid, config, max_jsd, max_exact_jsd)
Usage:  .venv/bin/python src/run_ablation.py [--runs 10]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probes  # noqa: E402
from qsim import unitary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

CONFIGS = {"A1": ("basis", "Z"), "A2": ("dep", "Z"), "A3": ("basis", "XYH"), "A4": ("dep", "XYH"),
           "S2": ("depcode", "Z"), "S4": ("depcode", "XYH")}
CFG_NAMES = list(CONFIGS)
K = probes.NUM_PROBES

_BUNDLE = None


def _load():
    global _BUNDLE
    if _BUNDLE is None:
        with open(ROOT / "data" / "mutants.pkl", "rb") as f:
            b = pickle.load(f)
        b["U0"] = {name: unitary(c["ops"], c["n"]) for name, c in b["originals"].items()}
        _BUNDLE = b
    return _BUNDLE


def run_one(run: int):
    b = _load()
    muts = b["mutants"]
    sampled = np.full((len(muts), len(CFG_NAMES), K), np.nan, dtype=np.float32)
    exact = np.full_like(sampled, np.nan)
    t0 = time.time()
    for mi, m in enumerate(muts):
        n, U0, U1 = m["n"], b["U0"][m["program"]], m["U"]
        rng = np.random.default_rng([run, m["mid"]])
        states = {fam: probes.INPUTS[fam](n, rng)[0] for fam in ("basis", "dep", "depcode")}
        for ci, cname in enumerate(CFG_NAMES):
            fam, pool = CONFIGS[cname]
            st = states[fam]
            k = len(st)
            meas_list = probes.measurement_choices(pool, k, rng)
            p_ref = probes.output_probs(U0, st, meas_list, n)
            p_mut = probes.output_probs(U1, st, meas_list, n)
            s_ref = probes.sample_dists(p_ref, rng)
            s_mut = probes.sample_dists(p_mut, rng)
            sampled[mi, ci, :k] = probes.jsd(s_ref, s_mut)
            exact[mi, ci, :k] = probes.jsd(p_ref, p_mut)
    print(f"run {run} done in {time.time() - t0:.1f}s", flush=True)
    return run, sampled, exact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--procs", type=int, default=10)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    muts = _load()["mutants"]
    with Pool(min(args.procs, args.runs)) as pool:
        results = sorted(pool.map(run_one, range(args.runs)), key=lambda r: r[0])
    sampled = np.stack([r[1] for r in results])
    exact = np.stack([r[2] for r in results])
    mids = np.array([m["mid"] for m in muts])
    np.savez_compressed(RAW / "probe_jsd.npz", sampled=sampled, exact=exact, mids=mids,
                        configs=np.array(CFG_NAMES))
    rows = []
    for r in range(args.runs):
        for ci, c in enumerate(CFG_NAMES):
            rows.append(pd.DataFrame({"run": r, "mid": mids, "config": c,
                                      "max_jsd": np.nanmax(sampled[r, :, ci], axis=1),
                                      "max_exact_jsd": np.nanmax(exact[r, :, ci], axis=1)}))
    pd.concat(rows).to_csv(RAW / "maxjsd.csv", index=False)
    print("saved", RAW / "probe_jsd.npz", RAW / "maxjsd.csv")


if __name__ == "__main__":
    main()
