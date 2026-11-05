"""Metrics and tables from results/preds/*.csv (contract in the research plan).

Usage:
    python -m src.evaluate               # writes results/tables/*.csv|tex and results/figures/robustness_curve.png
    python -m src.evaluate --selftest    # synthetic preds: perfect -> AUC 1, random -> AUC ~0.5

Main metric: cancer AUC (p_cancer = p_BCC + p_MEL + p_SCC) with 95% CI by patient bootstrap.
Operating point: threshold for 90% sensitivity fixed on val, applied unchanged to test.
"""
import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
from scipy.stats import binomtest, rankdata
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from src.data import CLASSES

CANCER = ["BCC", "MEL", "SCC"]
CANCER_IDX = [CLASSES.index(c) for c in CANCER]
SEEN = ["brightness", "gaussian_blur", "gaussian_noise", "hair"]
UNSEEN = ["motion_blur", "jpeg"]
MODEL_NAMES = {"effb0": "EfficientNet-B0", "vits16": "ViT-S/16", "baseline_logreg": "ResNet-50 + LR",
               "ham_effb0": "EfficientNet-B0 (HAM)"}
N_BOOT = 1000


def parse_run(run_id):
    m = re.fullmatch(r"(.+)_(none|standard|robust)_s(\d+)", run_id)
    return (m[1], m[2], int(m[3])) if m else (run_id, "-", 0)


def p_cancer(df):
    p = df[[f"p_{c}" for c in CANCER]].sum(axis=1).to_numpy()
    assert ((p >= -1e-6) & (p <= 1 + 1e-6)).all(), "p_cancer outside [0, 1]"
    return p


def threshold_at_sensitivity(val, target=0.9):
    """Largest threshold with sensitivity >= target. Only val rows are accepted."""
    assert (val["split"] == "val").all() and (val["corruption"] == "clean").all(), "threshold must come from clean val"
    pos = np.sort(p_cancer(val)[val["y_true"].isin(CANCER_IDX).to_numpy()])[::-1]
    return pos[int(np.ceil(target * len(pos))) - 1]


def boot_auc_ci(ys, scores, groups, seed=0):
    """95% CI of the mean AUC over runs (aligned rows), resampling patients with replacement."""
    rng = np.random.default_rng(seed)
    idx_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    stats = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_group[i] for i in rng.integers(len(idx_by_group), size=len(idx_by_group))])
        if ys[idx].min() == ys[idx].max():
            continue
        stats.append(np.mean([roc_auc_score(ys[idx], s[idx]) for s in scores]))
    return np.percentile(stats, [2.5, 97.5])


def fast_auc(y, scores):
    """AUC of each row of scores (..., n) against boolean y, via Mann-Whitney ranks (ties averaged)."""
    n1 = y.sum()
    return (rankdata(scores, axis=-1)[..., y].sum(-1) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def condition_scores(test, img_ids):
    """p_cancer as (conditions, images): clean first, then each corruption x severity 1..5, aligned to img_ids."""
    pc = test.assign(pc=p_cancer(test)).set_index(["corruption", "severity", "img_id"])["pc"]
    conds = [("clean", 0)] + [(c, s) for c in SEEN + UNSEEN for s in range(1, 6)]
    out = np.stack([pc.loc[c].reindex(img_ids).to_numpy() for c in conds])
    assert not np.isnan(out).any(), "missing test rows"
    return out


def boot_paired(y, scores, groups, seed=0):
    """Clean AUC and mean corrupted AUC (seed means) per key of scores {key: (seeds, conditions, n)}.

    Returns the point estimate and N_BOOT patient resamples; every key sees the same resamples (paired)."""
    def stat(idx):
        out = {}
        for k, s in scores.items():
            auc = fast_auc(y[idx], s[..., idx]).mean(0)
            out[k] = np.array([auc[0], auc[1:].mean()])
        return out

    rng = np.random.default_rng(seed)
    idx_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_group[i] for i in rng.integers(len(idx_by_group), size=len(idx_by_group))])
        if y[idx].min() != y[idx].max():
            boots.append(stat(idx))
    return stat(np.arange(len(y))), boots


def clopper_pearson(k, n):
    ci = binomtest(k, n).proportion_ci(method="exact")
    return ci.low, ci.high


def evaluate_run(preds):
    """Per-run metrics on clean test + AUC per (corruption, severity)."""
    val = preds[preds["split"] == "val"]
    test = preds[preds["split"] == "test"]
    thr = threshold_at_sensitivity(val)
    clean = test[test["corruption"] == "clean"]
    y = clean["y_true"].to_numpy()
    is_cancer = np.isin(y, CANCER_IDX)
    flag = p_cancer(clean) >= thr
    probs = clean[[f"p_{c}" for c in CLASSES]].to_numpy()
    row = dict(threshold=thr, auc=roc_auc_score(is_cancer, p_cancer(clean)),
               sens=flag[is_cancer].mean(), spec=(~flag[~is_cancer]).mean(),
               bacc=balanced_accuracy_score(y, probs.argmax(1)), macro_f1=f1_score(y, probs.argmax(1), average="macro"))
    for c in CANCER:
        sub = y == CLASSES.index(c)
        k, n = int(flag[sub].sum()), int(sub.sum())
        row.update({f"sens_{c}": k / n, f"k_{c}": k, f"n_{c}": n})
        row[f"sens_{c}_lo"], row[f"sens_{c}_hi"] = clopper_pearson(k, n)
    auc = (test.assign(pc=p_cancer(test), yc=test["y_true"].isin(CANCER_IDX))
           .groupby(["corruption", "severity"])[["yc", "pc"]]
           .apply(lambda g: roc_auc_score(g["yc"], g["pc"])).rename("auc").reset_index())
    return row, auc


def pm(s, digits=3):
    return f"{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}" if len(s) > 1 else f"{s.mean():.{digits}f}"


def label(model, aug):
    return MODEL_NAMES.get(model, model), aug


def write_tex(df, path, caption, wide=False):
    """LaTeX table that shrinks to the column (or, if wide, to the page width in a table*); needs adjustbox."""
    df = df.map(lambda v: v.replace("±", "$\\pm$").replace("%", "\\%") if isinstance(v, str) else v)
    df.columns = [c.replace("%", "\\%") for c in df.columns]
    tex = df.to_latex(index=False, caption=caption, label=f"tab:{Path(path).stem}", position="t",
                      column_format="l" * 2 + "c" * (df.shape[1] - 2))
    tex = (tex.replace("\\begin{tabular}", "\\centering\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}")
           .replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}"))
    Path(path).write_text(tex.replace("{table}", "{table*}") if wide else tex)


def main(preds_dir="results/preds", split_path="data/splits/pad_split.csv", tables="results/tables",
         figures="results/figures", logs_dir="results/logs", meta_path="data/raw/pad_256/metadata.csv"):
    Path(tables).mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(split_path)
    patients = split.set_index("img_id")["patient_id"]

    # Dataset summary per partition (images, patients, classes).
    data = pd.crosstab(split["split"], split["diagnostic"]).reindex(["train", "val", "test"]).reindex(columns=CLASSES, fill_value=0)
    data.insert(0, "Pacientes", split.groupby("split")["patient_id"].nunique())
    data.insert(0, "Imágenes", data[CLASSES].sum(1))
    data["Cáncer"] = data[["BCC", "MEL", "SCC"]].sum(1)
    data.index = data.index.map({"train": "Entrenamiento", "val": "Validación", "test": "Prueba"})
    data = data.rename_axis("Partición").reset_index().astype(str)
    data.to_csv(f"{tables}/data.csv", index=False)
    write_tex(data, f"{tables}/data.tex", "Resumen de PAD-UFES-20 por partición (agrupada por paciente). "
              "Cáncer = BCC + MEL + SCC.")
    runs, aucs, clean_test, test_all = [], [], {}, {}
    for f in sorted(Path(preds_dir).glob("*.csv")):
        preds = pd.read_csv(f)
        model, aug, seed = parse_run(f.stem)
        row, auc = evaluate_run(preds)
        runs.append(dict(run_id=f.stem, model=model, aug=aug, seed=seed, **row))
        aucs.append(auc.assign(run_id=f.stem, model=model, aug=aug, seed=seed))
        clean_test[f.stem] = preds.query("split == 'test' and corruption == 'clean'").sort_values("img_id")
        test_all[f.stem] = preds[preds["split"] == "test"]
    runs, aucs = pd.DataFrame(runs), pd.concat(aucs, ignore_index=True)
    runs.to_csv(f"{tables}/per_run.csv", index=False, float_format="%.4f")

    # Main table: clean test, mean +- sd over seeds, CI of the seed-mean AUC by patient bootstrap.
    order = {"none": 0, "standard": 1, "robust": 2, "-": 3}
    groups = sorted(runs.groupby(["model", "aug"]).groups, key=lambda g: (g[0], order[g[1]]))
    main_rows, six_rows = [], []
    for model, aug in groups:
        g = runs[(runs["model"] == model) & (runs["aug"] == aug)]
        frames = [clean_test[r] for r in g["run_id"]]
        assert all((fr["img_id"].to_numpy() == frames[0]["img_id"].to_numpy()).all() for fr in frames)
        y = frames[0]["y_true"].isin(CANCER_IDX).to_numpy()
        lo, hi = boot_auc_ci(y, [p_cancer(fr) for fr in frames], patients.loc[frames[0]["img_id"]].to_numpy())
        name, aug_name = label(model, aug)
        main_rows.append({"Modelo": name, "Augmentation": aug_name, "Semillas": len(g),
                          "AUC cáncer": pm(g["auc"]), "IC 95 %": f"[{lo:.3f}, {hi:.3f}]",
                          "Espec. @ sens. 90 %": pm(g["spec"]), "Sens. test": pm(g["sens"]),
                          "Sens. BCC": pm(g["sens_BCC"], 2), "Sens. SCC": pm(g["sens_SCC"], 2),
                          "Sens. MEL": pm(g["sens_MEL"], 2)})
        six_rows.append({"Modelo": name, "Augmentation": aug_name, "BACC": pm(g["bacc"]), "Macro-F1": pm(g["macro_f1"])})
    main_df, six_df = pd.DataFrame(main_rows), pd.DataFrame(six_rows)
    main_df.to_csv(f"{tables}/main.csv", index=False)
    six_df.to_csv(f"{tables}/main_6class.csv", index=False)
    best = main_df.copy()  # bold the best mean in the LaTeX table only
    for col in ("AUC cáncer", "Espec. @ sens. 90 %"):
        i = best[col].str.split().str[0].astype(float).idxmax()
        best.loc[i, col] = f"\\textbf{{{best.loc[i, col]}}}"
    write_tex(best.iloc[:, [0, 1, 3, 4, 5, 6]], f"{tables}/main.tex",
              "Triaje cáncer vs. no cáncer en el test limpio de PAD-UFES-20 (media $\\pm$ sd entre semillas; "
              "IC 95 \\% por bootstrap de pacientes). Umbral fijado en validación a sensibilidad 90 \\%.", wide=True)
    write_tex(six_df, f"{tables}/main_6class.tex", "Clasificación en 6 clases en el test limpio (media $\\pm$ sd).")

    # Robustness: relative AUC drop per seed, then mean +- sd over seeds.
    clean_auc = aucs[aucs["corruption"] == "clean"].set_index("run_id")["auc"]
    aucs["drop"] = (aucs["run_id"].map(clean_auc) - aucs["auc"]) / aucs["run_id"].map(clean_auc)
    curve = aucs.groupby(["model", "aug", "corruption", "severity"])[["auc", "drop"]].agg(["mean", "std"])
    curve.columns = ["_".join(c) for c in curve.columns]
    curve.reset_index().to_csv(f"{tables}/robustness.csv", index=False, float_format="%.4f")

    corrupted = aucs[aucs["corruption"] != "clean"]
    per_seed = corrupted.groupby(["model", "aug", "seed", "corruption"])["drop"].mean().unstack()
    per_seed["Vistas"] = per_seed[SEEN].mean(axis=1)
    per_seed["No vistas"] = per_seed[UNSEEN].mean(axis=1)
    rob_rows = []
    for model, aug in groups:
        g = per_seed.loc[(model, aug)]
        rob_rows.append({"Modelo": label(model, aug)[0], "Augmentation": aug,
                         **{c: pm(100 * g[c], 1) for c in SEEN + UNSEEN + ["Vistas", "No vistas"]}})
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(f"{tables}/robustness_drop.csv", index=False)
    write_tex(rob_df.rename(columns=lambda c: c.replace("_", " ")), f"{tables}/robustness.tex",
              "Caída relativa de la AUC cáncer (\\%) bajo corrupción, promedio de las severidades 1--5 "
              "(media $\\pm$ sd entre semillas). motion blur y jpeg no se usan en la augmentation.", wide=True)

    # Paired differences (H1: aug vs aug within a model; H2: model vs model with the same aug).
    first = clean_test[runs["run_id"].iloc[0]]
    img_ids, y_test = first["img_id"].to_numpy(), first["y_true"].isin(CANCER_IDX).to_numpy()
    scores = {k: np.stack([condition_scores(test_all[r], img_ids) for r in g["run_id"]])
              for k, g in runs.groupby(["model", "aug"])}
    pairs = [((m2, a), (m1, a)) for a in order
             for m1, m2 in combinations(sorted(m for m, aa in scores if aa == a), 2)
             if "ham_" not in m1 + m2 or m1.removeprefix("ham_") == m2.removeprefix("ham_")]  # H3: HAM only vs. its PAD twin
    pairs += [((m, a2), (m, a1)) for m in sorted({m for m, _ in scores})
              for a1, a2 in combinations(["none", "standard", "robust"], 2) if (m, a1) in scores and (m, a2) in scores]
    full, boots = boot_paired(y_test, scores, patients.loc[img_ids].to_numpy())
    paired_rows = []
    for a, b in pairs:
        d = np.array([bt[a] - bt[b] for bt in boots])
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        point = full[a] - full[b]
        name = (f"{label(*a)[0]} - {label(*b)[0]} ({a[1]})" if a[0] != b[0] else f"{label(*a)[0]}: {a[1]} - {b[1]}")
        paired_rows.append({"Comparación": name, "Dif. AUC limpia": f"{point[0]:+.3f}",
                            "IC 95 % (limpia)": f"[{lo[0]:+.3f}, {hi[0]:+.3f}]",
                            "Dif. AUC con corrupción": f"{point[1]:+.3f}",
                            "IC 95 % (corrupción)": f"[{lo[1]:+.3f}, {hi[1]:+.3f}]"})
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(f"{tables}/paired.csv", index=False)
    write_tex(paired_df, f"{tables}/paired.tex",
              "Diferencias emparejadas de AUC cáncer en test (media de semillas de cada grupo). Con corrupción: "
              "media de las 30 condiciones (6 corrupciones $\\times$ 5 severidades). IC 95 \\% por bootstrap de "
              "pacientes con los mismos remuestreos para ambos grupos.", wide=True)

    # Sensitivity per cancer subtype at the operating threshold; exact CI with the seed-mean hit count.
    sub_rows = []
    for model, aug in groups:
        g = runs[(runs["model"] == model) & (runs["aug"] == aug)]
        row = {"Modelo": label(model, aug)[0], "Augmentation": aug}
        for c in CANCER:
            n, k = int(g[f"n_{c}"].iloc[0]), int(round(g[f"k_{c}"].mean()))
            lo, hi = clopper_pearson(k, n)
            row[f"{c} (n={n})"] = f"{pm(g[f'sens_{c}'], 2)} [{lo:.2f}, {hi:.2f}]"
        sub_rows.append(row)
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(f"{tables}/subtype.csv", index=False)
    write_tex(sub_df, f"{tables}/subtype.tex",
              "Sensibilidad por subtipo de cáncer en el test limpio al umbral de operación (media $\\pm$ sd entre "
              "semillas). IC 95 \\% exacto de Clopper-Pearson con el número medio de aciertos, redondeado.", wide=True)

    # Per-run flags at the operating threshold (rows aligned with `first`, sorted by img_id).
    thr = runs.set_index("run_id")["threshold"]
    pc = {r: p_cancer(clean_test[r]) for r in runs["run_id"]}
    is_mel = (first["y_true"] == CLASSES.index("MEL")).to_numpy()

    # Melanoma cases: mean p_cancer over seeds and how many seeds refer each one.
    meta = pd.read_csv(meta_path).set_index("img_id")
    cases = meta.loc[img_ids[is_mel], ["age", "region", "fitspatrick", "diameter_1", "diameter_2"]].reset_index()
    for model, aug in groups:
        rids = runs.loc[(runs["model"] == model) & (runs["aug"] == aug), "run_id"]
        p = np.stack([pc[r][is_mel] for r in rids])
        hits = np.stack([pc[r][is_mel] >= thr[r] for r in rids]).sum(0)
        cases[f"{label(model, aug)[0]} {aug}"] = [f"{m:.2f} ({h}/{len(rids)})" for m, h in zip(p.mean(0), hits)]
    cases.to_csv(f"{tables}/melanoma_cases.csv", index=False)

    # Phototype: Fitzpatrick is missing only in ACK/NEV/SEK, so "unknown" has no cancers (no AUC, only specificity).
    fitz = meta.loc[img_ids, "fitspatrick"].to_numpy()
    photo = np.where(np.isnan(fitz), "Desconocido", np.where(fitz <= 2, "I-II", "III-VI"))
    bias_rows = []
    for model, aug in groups:
        rids = runs.loc[(runs["model"] == model) & (runs["aug"] == aug), "run_id"]
        for ph in ("I-II", "III-VI", "Desconocido"):
            m = photo == ph
            yc = y_test[m]
            row = {"Modelo": label(model, aug)[0], "Augmentation": aug, "Fototipo": ph,
                   "n (cáncer/no)": f"{yc.sum()}/{(~yc).sum()}"}
            # ponytail: AUC only with >= 10 per class; below that a single image swings it by > 0.05
            ok = min(yc.sum(), (~yc).sum()) >= 10
            row["AUC cáncer"] = pm(pd.Series([roc_auc_score(yc, pc[r][m]) for r in rids])) if ok else "-"
            row["Sens."] = pm(pd.Series([(pc[r][m][yc] >= thr[r]).mean() for r in rids])) if yc.any() else "-"
            row["Espec."] = pm(pd.Series([(pc[r][m][~yc] < thr[r]).mean() for r in rids])) if (~yc).any() else "-"
            bias_rows.append(row)
    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(f"{tables}/bias.csv", index=False)
    write_tex(bias_df, f"{tables}/bias.tex",
              "Triaje por fototipo de Fitzpatrick en el test limpio (media $\\pm$ sd entre semillas; umbral de "
              "operación fijado en validación). El fototipo falta solo en lesiones no cancerosas. "
              "AUC solo con al menos 10 casos por clase.", wide=True)

    # Confusion matrices (6 classes, robust augmentation, summed over seeds, row-normalized).
    robust = [m for m, a in groups if a == "robust"]
    if robust:
        fig, axes = plt.subplots(1, len(robust), figsize=(5 * len(robust), 4.4), squeeze=False)
        for ax, model in zip(axes[0], robust):
            cm = sum(confusion_matrix(clean_test[r]["y_true"], clean_test[r][[f"p_{c}" for c in CLASSES]]
                                      .to_numpy().argmax(1), labels=range(len(CLASSES)))
                     for r in runs.loc[(runs["model"] == model) & (runs["aug"] == "robust"), "run_id"])
            cmn = cm / cm.sum(1, keepdims=True)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i, j in np.ndindex(cm.shape):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
            ax.set_xticks(range(len(CLASSES)), CLASSES)
            ax.set_yticks(range(len(CLASSES)), CLASSES)
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Real")
            ax.set_title(f"{label(model, 'robust')[0]} · robust")
        plt.tight_layout()
        plt.savefig(f"{figures}/confusion_robust.png", dpi=200)
        plt.close(fig)

    # Compute cost, from the training logs (wall time includes the final 31-condition prediction).
    logs = [json.load(open(p)) for p in sorted(Path(logs_dir).glob("*.json"))]
    cost_df = None
    if logs:
        cost = pd.DataFrame([dict(zip(("model", "aug", "seed"), parse_run(lg["run_id"])), arch=lg["config"]["model"],
                                  gpu=lg["gpu"], epochs=lg["epochs_run"], best=lg["best_epoch"],
                                  minutes=lg["wall_seconds"] / 60) for lg in logs])
        cost_rows = []
        for (model, aug), g in sorted(cost.groupby(["model", "aug"]), key=lambda kv: (kv[0][0], order[kv[0][1]])):
            params = sum(p.numel() for p in timm.create_model(g["arch"].iloc[0], num_classes=len(CLASSES)).parameters())
            cost_rows.append({"Modelo": label(model, aug)[0], "Augmentation": aug, "GPU": ", ".join(g["gpu"].unique()),
                              "Parámetros (M)": f"{params / 1e6:.1f}", "Corridas": len(g),
                              "Épocas": pm(g["epochs"], 1), "Mejor época": pm(g["best"], 1),
                              "Min por corrida": pm(g["minutes"], 1), "Horas totales": f"{g['minutes'].sum() / 60:.2f}"})
        cost_rows.append({"Modelo": "Total", "Corridas": len(cost), "Horas totales": f"{cost['minutes'].sum() / 60:.2f}"})
        cost_df = pd.DataFrame(cost_rows).fillna("")
        cost_df.to_csv(f"{tables}/cost.csv", index=False)
        write_tex(cost_df, f"{tables}/cost.tex",
                  "Costo computacional por condición (media $\\pm$ sd entre semillas). Los minutos incluyen la "
                  "predicción final sobre validación y las 31 condiciones de test.", wide=True)

    # Figure: AUC vs severity, one panel per corruption, one line per (model, aug).
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    styles = {"none": "C0", "standard": "C1", "robust": "C2", "-": "C7"}
    for ax, corr in zip(axes.flat, SEEN + UNSEEN):
        for (model, aug) in groups:
            clean = curve.loc[(model, aug, "clean", 0)]
            c = curve.loc[(model, aug, corr)]
            x = [0] + list(c.index)
            ax.errorbar(x, [clean["auc_mean"]] + list(c["auc_mean"]),
                        yerr=[0 if np.isnan(clean["auc_std"]) else clean["auc_std"]] + list(c["auc_std"].fillna(0)),
                        color=styles[aug], ls={"vits16": "--", "ham_effb0": ":"}.get(model, "-"), marker="o", ms=3, capsize=2,
                        label=f"{label(model, aug)[0]} · {aug}")
        ax.set_title(corr + ("" if corr in SEEN else " (no vista)"))
        ax.set_xticks(range(6))
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Severidad (0 = limpia)")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC cáncer (test)")
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{figures}/robustness_curve.png", dpi=200)

    print(main_df.iloc[:, :7].to_string(index=False))
    print(six_df.to_string(index=False))
    print(rob_df.to_string(index=False))
    print(paired_df.to_string(index=False))
    print(sub_df.to_string(index=False))
    print(bias_df.to_string(index=False))
    print(cases.to_string(index=False))
    if cost_df is not None:
        print(cost_df.to_string(index=False))


def selftest():
    """Perfect preds -> AUC 1; random preds -> AUC ~0.5; the threshold refuses test rows."""
    import tempfile

    split = pd.read_csv("data/splits/pad_split.csv").query("split != 'train'")
    y = split["diagnostic"].map(CLASSES.index).to_numpy()
    with tempfile.TemporaryDirectory() as d:
        Path(f"{d}/preds").mkdir()
        rng = np.random.default_rng(0)
        for name, probs in [("perfect", np.eye(len(CLASSES))[y]), ("random", rng.dirichlet(np.ones(6), len(y)))]:
            rows = [split.assign(corruption="clean", severity=0)]
            rows += [split[split["split"] == "test"].assign(corruption=c, severity=s) for c in SEEN + UNSEEN for s in range(1, 6)]
            df = pd.concat(rows, ignore_index=True)
            p = np.concatenate([probs] + [probs[split["split"].to_numpy() == "test"]] * 30)
            df = df[["img_id", "split", "corruption", "severity"]].assign(y_true=df["diagnostic"].map(CLASSES.index))
            df[[f"p_{c}" for c in CLASSES]] = p
            for s in (1, 2):
                df.to_csv(f"{d}/preds/{name}_none_s{s}.csv", index=False)
        Path(f"{d}/fig").mkdir()
        main(f"{d}/preds", tables=f"{d}/tables", figures=f"{d}/fig", logs_dir=f"{d}/logs")
        res = pd.read_csv(f"{d}/tables/per_run.csv").set_index("run_id")
        assert res.loc["perfect_none_s1", "auc"] == 1 and res.loc["perfect_none_s1", "bacc"] == 1
        assert abs(res.loc["random_none_s1", "auc"] - 0.5) < 0.1
        paired = pd.read_csv(f"{d}/tables/paired.csv").iloc[0]
        assert paired["Comparación"].startswith("random - perfect") and float(paired["Dif. AUC limpia"]) < -0.4
        assert float(paired["IC 95 % (limpia)"].strip("[]").split(",")[1]) < 0
        sub = pd.read_csv(f"{d}/tables/subtype.csv").set_index("Modelo")
        assert all(sub.loc["perfect"].iloc[1:].str.startswith("1.00 ± 0.00"))
        assert not Path(f"{d}/tables/cost.csv").exists()  # no logs -> no cost table
        test = df[df["split"] == "test"]
        yb, s = test["y_true"].isin(CANCER_IDX).to_numpy(), np.round(rng.random(len(test)), 1)  # with ties
        assert np.isclose(fast_auc(yb, s), roc_auc_score(yb, s))
        try:
            threshold_at_sensitivity(test)
            raise RuntimeError("threshold accepted test rows")
        except AssertionError:
            pass
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else main()
