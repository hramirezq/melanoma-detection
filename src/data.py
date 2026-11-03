"""PAD-UFES-20: metadata, patient-level split and Dataset.

Usage:
    python -m src.data --root data/raw/pad --out data/splits/pad_split.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
MALIGNANT = {"BCC", "MEL", "SCC"}


def load_pad(root):
    """Metadata with absolute image path, class index and malignancy flag."""
    root = Path(root)
    df = pd.read_csv(root / "metadata.csv")
    paths = {p.name: p for p in root.rglob("*.png")}
    df["path"] = df["img_id"].map(paths)
    assert df["path"].notna().all(), f"{df['path'].isna().sum()} images missing under {root}"
    df["label"] = df["diagnostic"].map(CLASSES.index)
    df["malignant"] = df["diagnostic"].isin(MALIGNANT).astype(int)
    return df


def make_split(df, seed=42):
    """~70/15/15 train/val/test, stratified by class, grouped by patient (no leakage)."""
    df = df.copy()
    test_fold = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(test_fold.split(df, df["label"], df["patient_id"]))
    rest = df.iloc[rest_idx]
    val_fold = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_idx, val_idx = next(val_fold.split(rest, rest["label"], rest["patient_id"]))
    df["split"] = "train"
    df.loc[df.index[test_idx], "split"] = "test"
    df.loc[rest.index[val_idx], "split"] = "val"
    patients = [set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "val", "test")]
    assert not (patients[0] & patients[1] or patients[0] & patients[2] or patients[1] & patients[2]), "patient leakage"
    return df


class PADDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/pad")
    ap.add_argument("--out", default="data/splits/pad_split.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = make_split(load_pad(args.root), args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df[["img_id", "patient_id", "lesion_id", "diagnostic", "split"]].to_csv(args.out, index=False)
    print(pd.crosstab(df["diagnostic"], df["split"], margins=True))
    print(df.groupby("split")["patient_id"].nunique().rename("patients"))
