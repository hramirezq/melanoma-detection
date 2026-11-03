"""Baseline: frozen ImageNet ResNet-50 embeddings -> LogisticRegression(class_weight="balanced").

Usage:
    python -m src.baseline    # writes results/preds/baseline_logreg.csv (same contract as src.train)
"""
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

from src.corruptions import CORRUPTIONS, build_transform
from src.data import CLASSES, load_pad
from src.train import CorruptedDataset, Net


def main(root="data/raw/pad_256", split="data/splits/pad_split.csv", out="results/preds/baseline_logreg.csv"):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    df = load_pad(root).merge(pd.read_csv(split)[["img_id", "split"]], on="img_id")
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}

    model = Net("resnet50").to(device)
    model.backbone.reset_classifier(0)  # forward now returns the 2048-d pooled embedding
    tf = build_transform("none", False)
    conditions = [("train", "clean", 0), ("val", "clean", 0), ("test", "clean", 0)]
    conditions += [("test", c, s) for c in CORRUPTIONS for s in range(1, 6)]
    dataset = ConcatDataset([CorruptedDataset(parts[sp], c, s, tf) for sp, c, s in conditions])
    model.eval()
    with torch.no_grad():
        feats = torch.cat([model(x.to(device)).cpu() for x, _ in DataLoader(dataset, batch_size=64, num_workers=4)]).numpy()

    meta = pd.concat([pd.DataFrame(dict(img_id=parts[sp]["img_id"].to_numpy(), split=sp, corruption=c, severity=s,
                                        y_true=parts[sp]["label"].to_numpy())) for sp, c, s in conditions],
                     ignore_index=True)
    is_train = (meta["split"] == "train").to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
    clf.fit(feats[is_train], meta.loc[is_train, "y_true"])
    assert list(clf.classes_) == list(range(len(CLASSES)))

    preds = meta[~is_train].reset_index(drop=True)
    preds[[f"p_{c}" for c in CLASSES]] = clf.predict_proba(feats[~is_train])
    preds.to_csv(out, index=False, float_format="%.6f")
    print(f"{out}: {len(preds)} rows")


if __name__ == "__main__":
    main()
