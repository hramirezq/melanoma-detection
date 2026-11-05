"""Local web demo: upload a skin-lesion photo -> cancer / no cancer. Research tool, not a medical device.

Usage:
    python -m app [--run vits16_robust_s1] [--port 7860]    # needs results/checkpoints/<run>.pt

The image is reduced to short side 256 px like the training data (src.data --resize), can be degraded with one
of the test corruptions to show robustness live, and is classified at the 90 %-sensitivity threshold fixed on
val (results/tables/per_run.csv).
"""
import argparse
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from src.corruptions import CORRUPTIONS, apply_corruption, build_transform
from src.data import CLASSES
from src.explain import gradcam, load, rollout

DISCLAIMER = ("Herramienta de investigación. **No es un dispositivo médico** ni sustituye el diagnóstico de un "
              "especialista. Entrenada con fotos clínicas de Brasil (PAD-UFES-20), mayoría de fototipos I–III.")
# The softmax always splits 100 % among the 6 trained classes: anything else gets a meaningless answer.
SCOPE = ("**Solo distingue 6 lesiones:** queratosis actínica, carcinoma basocelular, melanoma, nevo, carcinoma "
         "espinocelular y queratosis seborreica. Con otras condiciones (acné, eczema, picaduras, verrugas…) o fotos "
         "sin una lesión, el resultado no tiene sentido: el modelo no puede responder «ninguna de estas».")


def to_256(img):
    s = 256 / min(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS) if s < 1 else img


def build(run, device):
    model = load(run, device)
    thr = float(pd.read_csv("results/tables/per_run.csv").set_index("run_id").loc[run, "threshold"])
    method = rollout if run.startswith("vits16") else gradcam
    tf = build_transform("none", False)

    def predict(img, corruption, severity):
        if img is None:
            return None, None, None
        img = to_256(img.convert("RGB"))
        if corruption != "ninguna":
            img = Image.fromarray(apply_corruption(np.asarray(img), corruption, int(severity), "demo"))
        x = tf(img)[None].to(device)
        m, p = method(model, x)
        with torch.no_grad():  # ponytail: second forward pass (~20 ms) instead of returning logits from explain
            probs = torch.softmax(model(x), 1)[0].cpu().numpy()
        p = float(p)
        verdict = "CÁNCER: derivar a dermatología" if p >= thr else "NO CÁNCER"
        text = (f"## {verdict}\nProbabilidad de cáncer (BCC + MEL + SCC): **{p:.2f}** · umbral {thr:.2f} "
                f"(sensibilidad 90 % fijada en validación)\n\n{SCOPE}\n\n{DISCLAIMER}")
        m = F.interpolate(m.detach().cpu()[None], size=224, mode="bilinear")[0, 0].numpy()
        base = np.asarray(T.CenterCrop(224)(T.Resize(224)(img)), dtype=np.float32) / 255  # what the model sees
        overlay = 0.55 * base + 0.45 * matplotlib.colormaps["jet"](m)[..., :3]
        return text, {c: float(v) for c, v in zip(CLASSES, probs)}, (overlay * 255).astype(np.uint8)

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="vits16_robust_s1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    if not Path(f"results/checkpoints/{args.run}.pt").exists():
        raise SystemExit(f"falta results/checkpoints/{args.run}.pt (ver README, Demo web)")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    predict = build(args.run, device)

    # Test images, one per class (ACK, BCC, MEL, NEV, SCC, SEK) + the melanoma almost every model misses.
    ids = ["PAT_1006_53_716", "PAT_684_1302_588", "PAT_936_1772_886", "PAT_1516_1765_530", "PAT_380_1540_959",
           "PAT_1701_3135_5", "PAT_966_1825_584"]
    examples = [[p, "ninguna", 3] for i in ids if (p := Path(f"data/raw/pad_256/{i}.png")).exists()]

    with gr.Blocks(title="Triaje de lesiones cutáneas") as demo:
        gr.Markdown(f"# Triaje de lesiones cutáneas: cáncer vs. no cáncer\nModelo `{args.run}`. {DISCLAIMER}\n\n{SCOPE}")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Foto de la lesión")
                corruption = gr.Dropdown(["ninguna", *CORRUPTIONS], value="ninguna",
                                         label="Degradar la foto (para probar robustez)")
                severity = gr.Slider(1, 5, value=3, step=1, label="Severidad")
                btn = gr.Button("Evaluar", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                probs = gr.Label(num_top_classes=6, label="Probabilidades (6 clases)")
                heat = gr.Image(label="Dónde mira el modelo")
        btn.click(predict, [img, corruption, severity], [verdict, probs, heat])
        if examples:
            gr.Examples(examples, [img, corruption, severity], label="Imágenes de test (una por clase)")
    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
