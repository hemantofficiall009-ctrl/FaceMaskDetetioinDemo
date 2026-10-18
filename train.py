"""
Step 2: Train a MobileNetV2 (transfer learning) 3-class mask classifier.

Handles heavy class imbalance (mask_weared_incorrect is rare) with a weighted loss
and selects the best epoch by macro-F1 rather than plain accuracy.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE = 128


def loaders(root, bs):
    train_tf = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ds = {
        "train": datasets.ImageFolder(Path(root) / "train", train_tf),
        "val": datasets.ImageFolder(Path(root) / "val", eval_tf),
        "test": datasets.ImageFolder(Path(root) / "test", eval_tf),
    }
    dl = {k: DataLoader(v, batch_size=bs, shuffle=(k == "train"), num_workers=0) for k, v in ds.items()}
    return ds, dl


def run_eval(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += model(x.to(device)).argmax(1).cpu().tolist()
            ys += y.tolist()
    return ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/faces")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--freeze_epochs", type=int, default=3, help="train only the head first")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, dl = loaders(args.data, args.bs)
    classes = ds["train"].classes
    print("Classes:", classes, "| device:", device)

    # Inverse-frequency class weights
    counts = torch.bincount(torch.tensor(ds["train"].targets), minlength=len(classes)).float()
    weights = (counts.sum() / (len(classes) * counts)).to(device)
    print("Train counts:", counts.int().tolist())

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, len(classes)))
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    best_f1 = -1.0

    for epoch in range(args.epochs):
        frozen = epoch < args.freeze_epochs
        for p in model.features.parameters():
            p.requires_grad = not frozen
        lr = args.lr if frozen else args.lr / 10
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        model.train()
        total = 0.0
        for x, y in dl["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)

        ys, ps = run_eval(model, dl["val"], device)
        f1 = f1_score(ys, ps, average="macro")
        acc = sum(int(a == b) for a, b in zip(ys, ps)) / len(ys)
        print(f"Epoch {epoch+1:02d}/{args.epochs} {'(head only)' if frozen else '(fine-tune)'} "
              f"loss={total/len(ds['train']):.4f} val_acc={acc:.3f} val_macroF1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"state_dict": model.state_dict(), "classes": classes, "size": SIZE}, args.out)

    # Final test evaluation with the best checkpoint
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    ys, ps = run_eval(model, dl["test"], device)
    print("\n=== Test set ===")
    print(classification_report(ys, ps, target_names=classes, digits=3, zero_division=0))
    print(f"Saved best model to {args.out}")


if __name__ == "__main__":
    main()
