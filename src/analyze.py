"""Analyze the ablation results: metrics, thresholds, complementarity, structure, breakdowns, figures.

Inputs : results/raw/maxjsd.csv, results/raw/probe_jsd.npz, data/mutants.pkl, data/mutants_index.csv
Outputs: results/RESULTS.md (all tables, auto-generated), results/*.csv, results/fig_*.png
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAU = 0.0025
GRID = [1e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
MAIN = ["A1", "A2", "A3", "A4"]
ALL = MAIN + ["S2", "S4"]
LABEL = {"A1": "A1 basis + Z", "A2": "A2 DEP + Z", "A3": "A3 basis + X/Y/H", "A4": "A4 DEP + X/Y/H",
         "S2": "S2 DEPcode + Z", "S4": "S4 DEPcode + X/Y/H"}
PAPER = pd.DataFrame({  # QDoctor Table 2 (≤5-qubit subset, 10 runs)
    "config": MAIN, "FPR": [0, 0, 0, 0], "Recall": [85.280, 90.114, 92.011, 99.728],
    "Precision": [100, 100, 100, 100], "F1": [92.055, 94.800, 95.839, 99.864],
    "AUC": [0.909, 0.945, 0.953, 0.999]}).set_index("config")
COLORS = {"A1": "#2a78d6", "A2": "#eb6834", "A3": "#1baf7a", "A4": "#eda100"}
INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})


# ------------------------------------------------------------------ helpers
def confusion(kill: pd.Series, eqv: pd.Series) -> dict:
    tp = int((kill & ~eqv).sum()); fn = int((~kill & ~eqv).sum())
    fp = int((kill & eqv).sum()); tn = int((~kill & eqv).sum())
    rec = 100 * tp / (tp + fn) if tp + fn else np.nan
    prec = 100 * tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "FPR": 100 * fp / (fp + tn), "Recall": rec,
            "Precision": prec, "F1": f1}


def auc(df_run: pd.DataFrame) -> float:
    pts = [(0.0, 0.0), (1.0, 1.0)]
    for t in GRID:
        k = df_run.max_jsd > t
        c = confusion(k, df_run.equivalent)
        pts.append((c["FPR"] / 100, c["Recall"] / 100))
    pts = sorted(set(pts))
    xs, ys = zip(*pts)
    # for equal FPR keep the max TPR (upper envelope)
    env = pd.Series(ys, index=xs).groupby(level=0).max().cummax()
    return float(np.trapezoid(env.values, env.index.values))


def mean_sd(df: pd.DataFrame, cols, digits=3) -> pd.DataFrame:
    g = df.groupby("config")[cols]
    m, s = g.mean(), g.std(ddof=1)
    out = m.round(digits).astype(str) + " ± " + s.round(digits).astype(str)
    return out.reindex([c for c in ALL if c in out.index])


def md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(**kw) + "\n"


# ------------------------------------------------------------------ load
def load():
    d = pd.read_csv(RES / "raw" / "maxjsd.csv")
    idx = pd.read_csv(ROOT / "data" / "mutants_index.csv")
    with open(ROOT / "data" / "mutants.pkl", "rb") as f:
        bundle = pickle.load(f)
    sys.path.insert(0, str(ROOT / "src"))
    from qsim import unitary
    U0 = {name: unitary(c["ops"], c["n"]) for name, c in bundle["originals"].items()}
    struct = []
    for m in bundle["mutants"]:
        A, B = U0[m["program"]], m["U"]
        D = B @ A.conj().T
        off = np.abs(D - np.diag(np.diag(D))).max()
        struct.append({"mid": m["mid"],
                       "z_invisible": off < 1e-9,                       # U' = D U, D diagonal
                       "abs_invisible": np.abs(np.abs(A) - np.abs(B)).max() < 1e-9})
    idx = idx.merge(pd.DataFrame(struct), on="mid")
    idx["gate"] = idx.desc.str.extract(r"(?:->|add |remove )([a-z]+)")[0]
    d = d.merge(idx[["mid", "program", "algorithm", "n", "operator", "equivalent", "gate",
                     "z_invisible", "abs_invisible"]], on="mid")
    npz = np.load(RES / "raw" / "probe_jsd.npz")
    return d, idx, npz


# ------------------------------------------------------------------ analyses
def main_table(d):
    rows = []
    for (cfg, run), g in d.groupby(["config", "run"]):
        c = confusion(g.max_jsd > TAU, g.equivalent)
        c.update(config=cfg, run=run, AUC=auc(g))
        rows.append(c)
    per_run = pd.DataFrame(rows)
    per_run.to_csv(RES / "metrics_per_run.csv", index=False)
    return per_run


def tau_star(d):
    rows = []
    for (cfg, run), g in d.groupby(["config", "run"]):
        t = next(t for t in GRID if not (g[g.equivalent].max_jsd > t).any())
        c = confusion(g.max_jsd > t, g.equivalent)
        c.update(config=cfg, run=run, tau_star=t)
        rows.append(c)
    return pd.DataFrame(rows)


def wilcoxon_table(per_run):
    pairs = [("A2", "A1"), ("A3", "A1"), ("A4", "A2"), ("A4", "A3"), ("A3", "A2"), ("S4", "A4")]
    rows = []
    for a, b in pairs:
        for metric in ("Recall", "F1"):
            x = per_run[per_run.config == a].sort_values("run")[metric].values
            y = per_run[per_run.config == b].sort_values("run")[metric].values
            diff = x - y
            if np.allclose(diff, 0):
                p = 1.0
            else:
                p = wilcoxon(x, y, zero_method="zsplit", method="exact" if (diff != 0).all() else "auto").pvalue
            nz = diff[diff != 0]
            rrb = (np.sum(nz > 0) - np.sum(nz < 0)) / len(nz) if len(nz) else 0.0
            rows.append({"comparison": f"{a} vs {b}", "metric": metric,
                         "mean diff (pp)": round(diff.mean(), 3), "Wilcoxon p": round(p, 4),
                         "sign-consistency r": round(rrb, 2)})
    return pd.DataFrame(rows)


def complementarity(d):
    ieq = d[~d.equivalent]
    kill = ieq.assign(k=ieq.max_jsd > TAU).pivot_table(index=["run", "mid"], columns="config", values="k")
    kill = kill.astype(bool)
    rows = []
    for run, g in kill.groupby(level=0):
        m2, m3, m4, m1 = ~g.A2, ~g.A3, ~g.A4, ~g.A1
        rows.append({"run": run, "inequivalent": len(g),
                     "A1 misses": m1.sum(), "A2 misses": m2.sum(), "A3 misses": m3.sum(),
                     "missed by A2 only": (m2 & ~m3).sum(), "missed by A3 only": (m3 & ~m2).sum(),
                     "missed by both A2 and A3": (m2 & m3).sum(),
                     "A2 OR A3 recall (%)": 100 * (~(m2 & m3)).mean(),
                     "A4 misses": m4.sum(),
                     "A4 kills missed by both A2 and A3": (~m4 & m2 & m3).sum()})
    comp = pd.DataFrame(rows)
    summ = comp.drop(columns="run").agg(["mean", "std"]).T.round(2)
    rec = ieq.assign(k=ieq.max_jsd > TAU).groupby(["config", "run"]).k.mean().groupby("config").mean() * 100
    miss = 100 - rec
    inter = pd.DataFrame({
        "quantity": ["input gain  (A2 − A1)", "measurement gain  (A3 − A1)", "joint gain  (A4 − A1)",
                     "additive prediction  (A2−A1)+(A3−A1)", "interaction  (A4−A3)−(A2−A1)",
                     "miss-rate: A1, A2, A3, A4", "independent-effects prediction of A4 miss  (m2·m3/m1)"],
        "value": [f"{rec.A2 - rec.A1:+.2f} pp", f"{rec.A3 - rec.A1:+.2f} pp", f"{rec.A4 - rec.A1:+.2f} pp",
                  f"{(rec.A2 - rec.A1) + (rec.A3 - rec.A1):+.2f} pp",
                  f"{(rec.A4 - rec.A3) - (rec.A2 - rec.A1):+.2f} pp",
                  f"{miss.A1:.2f}%, {miss.A2:.2f}%, {miss.A3:.2f}%, {miss.A4:.2f}%",
                  f"{miss.A2 * miss.A3 / miss.A1:.2f}%  (observed {miss.A4:.2f}%)"]})
    return summ, inter


def structural(d):
    ieq = d[~d.equivalent].copy()
    ieq["category"] = np.select(
        [ieq.z_invisible, ieq.abs_invisible],
        ["Z-invisible (U' = D·U, D diagonal)", "|U'| = |U| but not Z-invisible"],
        "|U'| ≠ |U| (amplitude-changing)")
    ieq["kill"] = ieq.max_jsd > TAU
    rec = ieq.groupby(["category", "config"]).kill.mean().unstack()[ALL] * 100
    size = ieq[ieq.config == "A1"].groupby(["category", "run"]).size().groupby("category").mean()
    rec.insert(0, "mutants", size.astype(int))
    rec.insert(1, "share (%)", (100 * size / size.sum()).round(2))
    return rec.round(2)


def breakdown(d, by):
    ieq = d[~d.equivalent].assign(kill=lambda x: x.max_jsd > TAU)
    rec = ieq.groupby([by, "config"]).kill.mean().unstack()[ALL] * 100
    n_ieq = ieq[(ieq.config == "A1") & (ieq.run == 0)].groupby(by).size()
    n_eq = d[d.equivalent & (d.config == "A1") & (d.run == 0)].groupby(by).size()
    fp = d[d.equivalent].assign(kill=lambda x: x.max_jsd > TAU).groupby(by).kill.sum()
    rec.insert(0, "ineqv", n_ieq)
    rec.insert(1, "eqv", n_eq.reindex(rec.index).fillna(0).astype(int))
    rec["FP (all cfgs, all runs)"] = fp.reindex(rec.index).fillna(0).astype(int)
    return rec.round(2)


def gate_misses(d):
    ieq = d[~d.equivalent].assign(miss=lambda x: x.max_jsd <= TAU)
    t = ieq.groupby(["gate", "config"]).miss.mean().unstack()[MAIN] * 100
    t.insert(0, "ineqv mutants", ieq[(ieq.config == "A1") & (ieq.run == 0)].groupby("gate").size())
    return t.sort_values("A1", ascending=False).round(1)


def noise_floor(d):
    rows = []
    for cfg in ALL:
        e = d[(d.config == cfg) & d.equivalent].max_jsd
        i = d[(d.config == cfg) & ~d.equivalent]
        killed = i[i.max_jsd > TAU].max_jsd
        rows.append({"config": cfg, "eqv max-JSD: median": e.median(), "eqv: 99th pct": e.quantile(.99),
                     "eqv: max": e.max(), "τ": TAU, "ineqv killed: min max-JSD": killed.min(),
                     "ineqv: share with noise-free JSD = 0 (%)": 100 * (i.max_exact_jsd < 1e-12).mean()})
    return pd.DataFrame(rows).set_index("config")


def probe_budget(npz, d):
    sampled, mids, cfgs = npz["sampled"], npz["mids"], list(npz["configs"])
    eqv = d.drop_duplicates("mid").set_index("mid").loc[mids].equivalent.values
    ks = np.arange(1, sampled.shape[-1] + 1)
    curves = {}
    for c in ALL:
        ci = cfgs.index(c)
        arr = sampled[:, ~eqv, ci, :]                                   # run, mutant, probe
        curves[c] = [np.mean(np.nanmax(np.where(np.isnan(arr[..., :k]), -1, arr[..., :k]), axis=-1) > TAU) * 100
                     for k in ks]
    return ks, pd.DataFrame(curves, index=ks)


# ------------------------------------------------------------------ figures
def fig_recall(per_run):
    m = per_run.groupby("config").Recall.agg(["mean", "std"]).loc[MAIN]
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - w / 2 - 0.01, PAPER.Recall, w, color="#c3c2b7", label="QDoctor paper (≤5 qubits)")
    ax.bar(x + w / 2 + 0.01, m["mean"], w, yerr=m["std"], color=COLORS["A1"],
           label="This reproduction (3–4 qubits)", error_kw={"ecolor": INK2, "lw": 1})
    for i, c in enumerate(MAIN):
        ax.text(x[i] - w / 2, PAPER.Recall[c] + 0.8, f"{PAPER.Recall[c]:.1f}", ha="center", fontsize=8, color=INK2)
        ax.text(x[i] + w / 2, m["mean"][c] + 0.8, f"{m['mean'][c]:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, ["A1\nbasis + Z", "A2\nDEP + Z", "A3\nbasis + X/Y/H", "A4\nDEP + X/Y/H"])
    ax.set_ylabel("Recall (%)  at τ = 0.0025"); ax.set_ylim(60, 104)
    ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_title("Recall by probe configuration (FPR = 0 for every bar)", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(RES / "fig_recall_vs_paper.png", dpi=200); plt.close(fig)


def fig_budget(ks, curves):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for c in MAIN:
        ax.plot(ks, curves[c], color=COLORS[c], lw=2, marker="o", ms=3)
        ax.text(ks[-1] + 0.3, curves[c].iloc[-1], LABEL[c], va="center", fontsize=8, color=INK)
    ax.set_xlabel("Number of probes per mutant (k)"); ax.set_ylabel("Recall (%)  at τ = 0.0025")
    ax.set_xlim(0.5, 26.5); ax.set_xticks([1, 4, 8, 12, 16, 20])
    ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Recall as the probe budget grows (basis inputs stop at 2ⁿ probes)", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(RES / "fig_probe_budget.png", dpi=200); plt.close(fig)


def fig_heatmap(by_prog):
    t = by_prog[MAIN]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    im = ax.imshow(t.values, cmap="Blues", vmin=40, vmax=100, aspect="auto")
    ax.set_xticks(range(4), MAIN); ax.set_yticks(range(len(t)), t.index)
    for i in range(t.shape[0]):
        for j in range(4):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white" if v > 80 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("Recall (%)")
    ax.set_title("Recall by program and configuration", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(RES / "fig_recall_by_program.png", dpi=200); plt.close(fig)


def fig_separation(d):
    """Max-over-probes JSD per mutant and run, equivalent vs inequivalent, against tau."""
    bins = np.logspace(-6, 0, 49)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, MAIN):
        g = d[d.config == c]
        for eq, col, name in [(False, COLORS["A1"], "Inequivalent mutants"), (True, COLORS["A2"], "Equivalent mutants")]:
            v = g[g.equivalent == eq].max_jsd.clip(lower=1e-6)
            w = np.full(len(v), 100 / len(v))
            ax.hist(v, bins=bins, weights=w, histtype="step", lw=1.6, color=col, label=name)
        ax.axvline(TAU, color=INK2, lw=1, ls="--")
        ax.set_xscale("log"); ax.set_title(LABEL[c], fontsize=9, color=INK, loc="left")
        ax.yaxis.grid(True, color=GRIDC, lw=0.8); ax.set_axisbelow(True)
    axes[0, 0].text(TAU * 1.25, axes[0, 0].get_ylim()[1] * 0.92, "τ = 0.0025", fontsize=8, color=INK2, va="top")
    for ax in axes[1]:
        ax.set_xlabel("Largest JSD over a mutant's probes")
    fig.supylabel("Share of mutant-runs (%)", fontsize=10, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(RES / "fig_jsd_separation.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    d, idx, npz = load()
    per_run = main_table(d)
    cols = ["FPR", "Recall", "Precision", "F1", "AUC", "TP", "FP", "TN", "FN"]
    main_t = mean_sd(per_run, cols)
    main_t.index = [LABEL[c] for c in main_t.index]
    paper_t = PAPER.copy(); paper_t.index = [LABEL[c] for c in paper_t.index]
    ts = tau_star(d)
    ts_t = ts.groupby("config").agg(tau_star_min=("tau_star", "min"), tau_star_max=("tau_star", "max"),
                                    Recall=("Recall", "mean"), FPR=("FPR", "mean"), F1=("F1", "mean")).round(3)
    wil = wilcoxon_table(per_run)
    comp_summ, inter = complementarity(d)
    struct = structural(d)
    by_prog = breakdown(d, "program"); by_alg = breakdown(d, "algorithm"); by_op = breakdown(d, "operator")
    gm = gate_misses(d)
    nf = noise_floor(d)
    ks, curves = probe_budget(npz, d)
    for name, t in [("table_main", main_t), ("table_tau_star", ts_t), ("table_wilcoxon", wil),
                    ("table_complementarity", comp_summ), ("table_interaction", inter),
                    ("table_structural", struct), ("table_by_program", by_prog),
                    ("table_by_algorithm", by_alg), ("table_by_operator", by_op),
                    ("table_gate_misses", gm), ("table_noise_floor", nf), ("table_probe_budget", curves.round(2))]:
        t.to_csv(RES / f"{name}.csv")
    fig_recall(per_run); fig_budget(ks, curves); fig_heatmap(by_prog); fig_separation(d)

    n_eq = int(idx.equivalent.sum()); n_ieq = int((~idx.equivalent).sum())
    out = ["# RESULTS (auto-generated by `src/analyze.py` — do not edit by hand)\n",
           f"Mutants: {len(idx)} ({n_ieq} inequivalent, {n_eq} equivalent) from 8 programs; "
           f"10 runs; 20 probes/mutant (basis inputs: min(20, 2^n)); 20,000 shots; τ = {TAU} "
           "(base-2 JS divergence).\n",
           "## 1. Main ablation table (mean ± sd over 10 runs, τ = 0.0025; metrics in %)\n", md(main_t),
           "### Paper (QDoctor Table 2, ≤5-qubit subset)\n", md(paper_t),
           "## 2. τ* = smallest grid threshold with zero equivalent kills (per run)\n", md(ts_t),
           "## 3. Paired Wilcoxon signed-rank tests across the 10 runs\n", md(wil, index=False),
           "## 4. Complementarity of input-side and measurement-side gains (inequivalent mutants, per run mean / sd)\n",
           md(comp_summ), "\n### 2×2 factorial view (recall, mean over runs)\n", md(inter, index=False),
           "## 5. Recall by structural category of the mutant (noise-free property of U' vs U)\n", md(struct),
           "## 6. Noise floor: max-over-probes JSD of equivalent mutants vs τ\n", md(nf.round(6)),
           "## 7. Recall by program\n", md(by_prog), "### By algorithm\n", md(by_alg),
           "### By mutation operator\n", md(by_op),
           "## 8. Miss rate (%) by mutated gate (gate added / replaced-in / removed)\n", md(gm),
           "## 9. Recall vs probe budget k\n", md(curves.round(2).iloc[[0, 1, 3, 7, 11, 15, 19]]),
           "## Figures\n", "- `fig_recall_vs_paper.png`\n- `fig_probe_budget.png`\n- `fig_recall_by_program.png`\n- `fig_jsd_separation.png`\n"]
    (RES / "RESULTS.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
