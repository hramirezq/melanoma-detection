"""Train one run and write its log, checkpoint and predictions (contract in the research plan).

Usage:
    python -m src.train configs/effb0_standard_s1.yaml [--epochs 1] [--resume] [--out results]

Writes <out>/logs/<run_id>.json, <out>/checkpoints/<run_id>.pt (best epoch) and
<out>/preds/<run_id>.csv (val clean + test clean and 6 corruptions x 5 severities).
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.corruptions import CORRUPTIONS, apply_corruption, build_transform
from src.data import CLASSES, PADDataset, load_ham, load_pad

CANCER_IDX = [CLASSES.index(c) for c in ("BCC", "MEL", "SCC")]


class Net(nn.Module):
    """timm backbone that takes [0, 1] images and applies its own pretrained normalization."""

    def __init__(self, name, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=len(CLASSES))
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


class CorruptedDataset(Dataset):
    def __init__(self, df, corruption, severity, transform):
        self.df, self.corruption, self.severity, self.transform = df.reset_index(drop=True), corruption, severity, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = np.asarray(Image.open(row["path"]).convert("RGB"))
        if self.corruption != "clean":
            img = apply_corruption(img, self.corruption, self.severity, row["img_id"])
        return self.transform(Image.fromarray(img)), row["label"]


def p_cancer(probs):
    return probs[:, CANCER_IDX].sum(1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = [torch.softmax(model(x.to(device)).float(), 1).cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--epochs", type=int, help="override config (smoke test)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoints/<run_id>_last.pt")
    ap.add_argument("--root", default="data/raw/pad_256")
    ap.add_argument("--split", default="data/splits/pad_split.csv")
    ap.add_argument("--ham-root", default="data/raw/ham10000", help="only for configs with data: ham")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    run_id = Path(args.config).stem
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["epochs"] = args.epochs
    out = Path(args.out)
    for d in ("logs", "checkpoints", "preds"):
        (out / d).mkdir(parents=True, exist_ok=True)
    last_ckpt = out / "checkpoints" / f"{run_id}_last.pt"

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    amp = device == "cuda"

    df = load_pad(args.root).merge(pd.read_csv(args.split)[["img_id", "split"]], on="img_id")
    assert len(df) == 2298, len(df)
    train_df, val_df, test_df = (df[df["split"] == s] for s in ("train", "val", "test"))
    if cfg.get("data") == "ham":  # train and early-stop on HAM10000; PAD val/test are predicted as usual
        ham = load_ham(args.ham_root)
        train_df, fit_val_df = ham[ham["split"] == "train"], ham[ham["split"] == "val"]
    else:
        fit_val_df = val_df

    size = cfg["img_size"]
    loader_kw = dict(batch_size=cfg["batch_size"], num_workers=args.workers, pin_memory=amp,
                     persistent_workers=args.workers > 0)
    # Reseeded every epoch with (seed, epoch): shuffle and augmentation never replay after --resume.
    gen = torch.Generator()
    train_loader = DataLoader(PADDataset(train_df, build_transform(cfg["aug"], True, size)), shuffle=True,
                              drop_last=True, generator=gen, **{**loader_kw, "persistent_workers": False})
    val_loader = DataLoader(PADDataset(fit_val_df, build_transform(cfg["aug"], False, size)), **loader_kw)

    counts = np.bincount(train_df["label"], minlength=len(CLASSES))
    class_w = torch.tensor(len(train_df) / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    model = Net(cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler(enabled=amp)

    state = dict(epoch=0, history=[], best_auc=-1.0, best_epoch=0, bad_epochs=0, train_seconds=0.0)
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        print(f"resumed {run_id} at epoch {state['epoch']}")

    y_val = fit_val_df["label"].to_numpy()
    while state["epoch"] < cfg["epochs"] and state["bad_epochs"] < cfg["patience"]:
        t0 = time.time()
        gen.manual_seed(cfg["seed"] * 1000 + state["epoch"])
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()

        probs = predict(model, val_loader, device)
        val_auc = roc_auc_score(np.isin(y_val, CANCER_IDX), p_cancer(probs))
        val_bacc = balanced_accuracy_score(y_val, probs.argmax(1))
        state["epoch"] += 1
        state["train_seconds"] += time.time() - t0
        state["history"].append(dict(epoch=state["epoch"], train_loss=float(np.mean(losses)), val_auc_cancer=val_auc,
                                     val_bacc=val_bacc, lr=opt.param_groups[0]["lr"], seconds=time.time() - t0))
        if val_auc > state["best_auc"]:
            state.update(best_auc=val_auc, best_epoch=state["epoch"], bad_epochs=0)
            torch.save({"state_dict": model.state_dict(), "config": cfg, "run_id": run_id, "epoch": state["epoch"]},
                       out / "checkpoints" / f"{run_id}.pt")
        else:
            state["bad_epochs"] += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), state=state), last_ckpt)
        print(f"{run_id} ep {state['epoch']:2d}  loss {np.mean(losses):.3f}  val AUC cancer {val_auc:.3f}"
              f"  BACC {val_bacc:.3f}  {time.time() - t0:.0f}s", flush=True)

    # Final predictions with the best epoch (selected on val only).
    t0 = time.time()
    model.load_state_dict(torch.load(out / "checkpoints" / f"{run_id}.pt", map_location=device)["state_dict"])
    eval_tf = build_transform(cfg["aug"], False, size)
    conditions = [("val", val_df, "clean", 0), ("test", test_df, "clean", 0)]
    conditions += [("test", test_df, c, s) for c in CORRUPTIONS for s in range(1, 6)]
    # One loader over all conditions: workers are started once (spawn is slow on macOS).
    dataset = ConcatDataset([CorruptedDataset(part, c, s, eval_tf) for _, part, c, s in conditions])
    probs = predict(model, DataLoader(dataset, **{**loader_kw, "persistent_workers": False}), device)
    preds = pd.concat([pd.DataFrame(dict(img_id=part["img_id"].to_numpy(), split=split, corruption=c, severity=s,
                                         y_true=part["label"].to_numpy())) for split, part, c, s in conditions],
                      ignore_index=True)
    preds[[f"p_{c}" for c in CLASSES]] = probs
    preds.to_csv(out / "preds" / f"{run_id}.csv", index=False, float_format="%.6f")

    log = dict(run_id=run_id, config=cfg, history=state["history"], best_epoch=state["best_epoch"],
               best_val_auc_cancer=state["best_auc"], epochs_run=state["epoch"], device=device,
               gpu=torch.cuda.get_device_name(0) if device == "cuda" else f"{platform.machine()} ({device})",
               torch=torch.__version__, timm=timm.__version__, python=platform.python_version(),
               train_seconds=state["train_seconds"], predict_seconds=time.time() - t0,
               wall_seconds=state["train_seconds"] + time.time() - t0)
    json.dump(log, open(out / "logs" / f"{run_id}.json", "w"), indent=1)
    last_ckpt.unlink(missing_ok=True)  # Drive's FUSE mount can hide a file that was just overwritten
    print(f"{run_id}: best epoch {state['best_epoch']}  val AUC cancer {state['best_auc']:.3f}  "
          f"{log['wall_seconds'] / 60:.1f} min  -> {out}/preds/{run_id}.csv ({len(preds)} rows)")


if __name__ == "__main__":
    main()
