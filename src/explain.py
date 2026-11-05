"""Where the model looks: Grad-CAM (EfficientNet-B0) and attention rollout (ViT-S/16), clean vs. corrupted.

Usage:
    python -m src.explain    # needs results/checkpoints/{effb0,vits16}_{standard,robust}_s1.pt

Writes results/figures/gradcam_effb0.png, results/figures/rollout_vits16.png,
results/tables/explain_stability.csv|tex (correlation between the clean and corrupted maps over the test cancers)
and results/tables/inference.csv|tex (latency per image).
"""
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.corruptions import CORRUPTIONS, apply_corruption, build_transform
from src.data import CLASSES, load_pad
from src.evaluate import MODEL_NAMES, write_tex
from src.train import CANCER_IDX, CorruptedDataset, Net

FIG_CONDITIONS = [("clean", 0), ("gaussian_noise", 3), ("motion_blur", 3)]
SEVERITY = 3  # for the stability table


def log_p_cancer(logits):
    return torch.logsumexp(logits[:, CANCER_IDX], 1) - torch.logsumexp(logits, 1)


def gradcam(model, x):
    """Grad-CAM of log p_cancer on the last conv block (7x7), each map scaled to [0, 1]."""
    acts = {}
    hook = model.backbone.bn2.register_forward_hook(lambda m, i, o: acts.update(a=o))
    logits = model(x)
    hook.remove()
    score = log_p_cancer(logits)
    (g,) = torch.autograd.grad(score.sum(), acts["a"])  # eval mode: each sample only feeds its own score
    cam = F.relu((g.mean((2, 3), keepdim=True) * acts["a"]).sum(1))
    return cam / cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None], score.exp()


@torch.no_grad()
def rollout(model, x):
    """Attention rollout (Abnar & Zuidema 2020): CLS -> patch attention through all layers (14x14), scaled to [0, 1]."""
    attns, hooks = [], []
    for blk in model.backbone.blocks:
        blk.attn.fused_attn = False  # the fused kernel never materializes the attention matrix
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: attns.append(o)))
    logits = model(x)
    for h in hooks:
        h.remove()
    n = attns[0].shape[-1]
    eye = torch.eye(n, device=x.device)
    r = eye.expand(len(x), n, n)
    for a in attns:
        a = 0.5 * a.mean(1) + 0.5 * eye  # residual connection
        r = (a / a.sum(-1, keepdim=True)) @ r
    side = int((n - model.backbone.num_prefix_tokens) ** 0.5)
    m = r[:, 0, model.backbone.num_prefix_tokens:].reshape(-1, side, side)
    return m / m.flatten(1).amax(1)[:, None, None], log_p_cancer(logits).exp()


def crop(path, img_id, corruption, severity):
    """The image exactly as the model sees it (corruption, resize, center crop), for the overlays."""
    img = np.asarray(Image.open(path).convert("RGB"))
    if corruption != "clean":
        img = apply_corruption(img, corruption, severity, img_id)
    return T.CenterCrop(224)(T.Resize(224)(Image.fromarray(img)))


def load(run_id, device):
    ck = torch.load(f"results/checkpoints/{run_id}.pt", map_location=device)
    model = Net(ck["config"]["model"], pretrained=False).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model


def explain(model, method, df, corruption, severity, device):
    loader = DataLoader(CorruptedDataset(df, corruption, severity, build_transform("none", False)), batch_size=32)
    maps, probs = [], []
    for x, _ in loader:
        m, p = method(model, x.to(device))
        maps.append(m.detach().cpu())
        probs.append(p.detach().cpu())
    maps, probs = torch.cat(maps).numpy(), torch.cat(probs).numpy()
    assert np.isfinite(maps).all() and maps.min() >= 0 and maps.max() <= 1 + 1e-5
    return maps, probs


@torch.no_grad()
def ms_per_image(model, device, n=50):
    """Mean latency of one 224 px image (batch 1), after warm-up."""
    model = model.to(device)
    x = torch.rand(1, 3, 224, 224, device=device)
    sync = torch.mps.synchronize if device == "mps" else torch.cuda.synchronize if device == "cuda" else lambda: None
    for _ in range(5):
        model(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    sync()
    return (time.perf_counter() - t0) / n * 1000


def figure(maps, probs, images, rows, model_key, path):
    """Rows = images; columns = original + (condition x augmentation) overlays, titled with p_cancer."""
    import matplotlib.pyplot as plt

    cols = [(c, s, a) for c, s in FIG_CONDITIONS for a in ("standard", "robust")]
    fig, axes = plt.subplots(len(rows), len(cols) + 1, figsize=(1.9 * (len(cols) + 1), 2.0 * len(rows)))
    for i, (_, row) in enumerate(rows.iterrows()):
        axes[i, 0].imshow(images[("clean", 0)][i])
        axes[i, 0].set_ylabel(f"{row['diagnostic']}\n{row['img_id'][:-4]}", fontsize=7)
        for j, (c, s, a) in enumerate(cols, 1):
            m = F.interpolate(torch.tensor(maps[(a, c, s)][i])[None, None], size=224, mode="bilinear")[0, 0]
            axes[i, j].imshow(images[(c, s)][i])
            axes[i, j].imshow(m, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            head = f"{c if s == 0 else f'{c} {s}'}\n{a}\n" if i == 0 else ""
            axes[i, j].set_title(f"{head}p={probs[(a, c, s)][i]:.2f}", fontsize=7, pad=2)
    axes[0, 0].set_title("original", fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"{MODEL_NAMES[model_key]}: {'Grad-CAM' if model_key == 'effb0' else 'attention rollout'}"
                 " (p = probabilidad de cáncer)", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def main(root="data/raw/pad_256", split="data/splits/pad_split.csv"):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    df = load_pad(root).merge(pd.read_csv(split)[["img_id", "split"]], on="img_id")
    test = df[df["split"] == "test"].sort_values("img_id").reset_index(drop=True)
    # Figure rows: 2 distinct lesions each of MEL, BCC and NEV (first by img_id).
    rows = pd.concat([test[test["diagnostic"] == c].drop_duplicates("lesion_id").head(2) for c in ("MEL", "BCC", "NEV")])
    images = {(c, s): [crop(p, i, c, s) for p, i in zip(rows["path"], rows["img_id"])] for c, s in FIG_CONDITIONS}

    # Stability over cancer images only: for a benign lesion Grad-CAM of p_cancer is often empty by design.
    cancers = test[test["label"].isin(CANCER_IDX)]
    stability, latency = [], []
    for key, method, fig_name in (("effb0", gradcam, "gradcam_effb0"), ("vits16", rollout, "rollout_vits16")):
        maps, probs = {}, {}
        for aug in ("standard", "robust"):
            run = f"{key}_{aug}_s1"
            model = load(run, device)
            clean, _ = explain(model, method, cancers, "clean", 0, device)
            row = {"Modelo": MODEL_NAMES[key], "Método": "Grad-CAM" if key == "effb0" else "Attention rollout",
                   "Augmentation": aug}
            for c in CORRUPTIONS:
                corr, _ = explain(model, method, cancers, c, SEVERITY, device)
                a, b = clean.reshape(len(cancers), -1), corr.reshape(len(cancers), -1)
                a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
                r = (a * b).sum(1) / np.sqrt((a**2).sum(1) * (b**2).sum(1)).clip(1e-12)
                row[c] = float(r.mean())
            row["Media"] = float(np.mean([row[c] for c in CORRUPTIONS]))
            stability.append(row)
            for c, s in FIG_CONDITIONS:
                maps[(aug, c, s)], probs[(aug, c, s)] = explain(model, method, rows, c, s, device)
            print(run, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)})
            if aug == "robust":  # same architecture for both augs: time it once
                latency.append({"Modelo": MODEL_NAMES[key],
                                "Parámetros (M)": f"{sum(q.numel() for q in model.parameters()) / 1e6:.1f}",
                                f"ms/imagen ({device})": f"{ms_per_image(model, device):.1f}",
                                "ms/imagen (cpu)": f"{ms_per_image(model, 'cpu'):.1f}"})
        figure(maps, probs, images, rows, key, f"results/figures/{fig_name}.png")

    st = pd.DataFrame(stability)
    st.to_csv("results/tables/explain_stability.csv", index=False, float_format="%.3f")
    tex = st.copy()
    tex[list(CORRUPTIONS) + ["Media"]] = tex[list(CORRUPTIONS) + ["Media"]].map(lambda v: f"{v:.2f}")
    write_tex(tex.rename(columns=lambda c: c.replace("_", " ")), "results/tables/explain_stability.tex",
              f"Estabilidad de la explicación: correlación de Pearson entre el mapa limpio y el mapa con corrupción "
              f"(severidad {SEVERITY}), media sobre los {len(cancers)} cánceres de test (semilla 1). Un mapa vacío "
              "(sin evidencia de cáncer) cuenta como correlación 0. Solo es comparable entre augmentations del "
              "mismo modelo, porque los métodos difieren.", wide=True)
    print(st.to_string(index=False, float_format="%.3f"))
    lat = pd.DataFrame(latency)
    lat.to_csv("results/tables/inference.csv", index=False)
    write_tex(lat, "results/tables/inference.tex", f"Tiempo de inferencia por imagen (lote 1, 224 px, media de 50 "
              f"pasadas tras calentamiento) en {platform.platform(terse=True)} ({platform.machine()}).")
    print(lat.to_string(index=False))


if __name__ == "__main__":
    main()
