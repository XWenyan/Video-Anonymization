import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score

# ===== 你的工程模块 =====
import aux_code.params_privacy as params
import config as cfg

from aux_code.vispr_dl_edit import vispr_dataset
from torchvision.models import resnet50, ResNet50_Weights

def safe_collate(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    return torch.utils.data.dataloader.default_collate(batch)

def train_epoch(epoch, dataloader, model, criterion, optimizer):
    model.train()
    losses = []

    for i, data in enumerate(dataloader):

        if data is None:
            continue

        imgs, labels, _ = data
        imgs = imgs.cuda()
        labels = torch.from_numpy(np.asarray(labels)).float().cuda()

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 100 == 0:
            print(f"[Train] Epoch {epoch} | Batch {i} | Loss {np.mean(losses):.5f}")

    return np.mean(losses)


# =========================
# Validation
# =========================
def val_epoch(dataloader, model, criterion):
    model.eval()

    losses = []
    preds = []
    gts = []

    with torch.no_grad():
        for data in dataloader:
            if data is None:
                continue

            imgs, labels, _ = data
            imgs = imgs.cuda()
            labels = torch.from_numpy(np.asarray(labels)).float().cuda()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            losses.append(loss.item())

            preds.extend(outputs.cpu().numpy())
            gts.extend(labels.cpu().numpy())

    preds = np.asarray(preds)
    gts = np.asarray(gts)

    ap = average_precision_score(gts, preds, average=None)
    macro_ap = np.mean(ap)

    print(f"[VAL] Loss {np.mean(losses):.5f} | Macro AP {macro_ap:.5f}")

    return macro_ap


# =========================
# Main
# =========================
def main():

    run_id = "fb_pretrain_vispr"
    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    # ===== Model =====
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, params.num_pa)
    model.cuda()

    # ===== Loss & Optim =====
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ===== Data =====
    train_dataset = vispr_dataset('train', shuffle=True)
    val_dataset = vispr_dataset('test', shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=safe_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=safe_collate
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # ===== Training Loop =====
    best_ap = 0

    for epoch in range(1, 101):
        print(f"\n===== Epoch {epoch} =====")

        train_loss = train_epoch(epoch, train_loader, model, criterion, optimizer)

        val_ap = val_epoch(val_loader, model, criterion)

        # ===== Save best =====
        if val_ap > best_ap:
            best_ap = val_ap

            save_path = os.path.join(save_dir, f"fb_best_{val_ap:.4f}.pth")
            torch.save({
                "epoch": epoch,
                "fb_model_state_dict": model.state_dict()
            }, save_path)

            print(f"🔥 Best model saved at {save_path}")

        # ===== Temp save =====
        torch.save({
            "epoch": epoch,
            "fb_model_state_dict": model.state_dict()
        }, os.path.join(save_dir, "fb_latest.pth"))


if __name__ == "__main__":
    main()