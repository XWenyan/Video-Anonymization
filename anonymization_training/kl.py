import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from mnad_model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import convAE


# =====================================================
# 修改这里
# =====================================================

ORIG_ROOT = "/mnt/private/xwy/aed-mae/data/avenue/test/frames"

ANON_ROOT = "/mnt/private/xwy/SPAct/anonymization_training/fa_result/AP26"

MODEL_PATH = "/mnt/private/xwy/SPAct/anonymization_training/saved_models/train0513/ft_model_57.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T_LENGTH = 5

IMG_H = 256
IMG_W = 256

# =====================================================
# image transform
# =====================================================

transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
])

# =====================================================
# load model
# =====================================================

model = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.eval()
encoder = model.encoder

model = model.to(DEVICE)
model.eval()

encoder = model.encoder

# =====================================================
# collect all videos
# =====================================================

video_dirs = sorted([
    d for d in os.listdir(ORIG_ROOT)
    if os.path.isdir(os.path.join(ORIG_ROOT, d))
])

# =====================================================
# metrics
# =====================================================

all_kl = []
all_cos = []

eps = 1e-6

# =====================================================
# evaluation
# =====================================================

for video in tqdm(video_dirs):

    orig_video_dir = os.path.join(ORIG_ROOT, video)
    anon_video_dir = os.path.join(ANON_ROOT, video)

    if not os.path.exists(anon_video_dir):
        print("Missing:", anon_video_dir)
        continue

    frames = sorted(os.listdir(orig_video_dir))

    if len(frames) < T_LENGTH - 1:
        continue

    for idx in range(T_LENGTH - 1, len(frames)):

        orig_imgs = []
        anon_imgs = []

        valid = True

        for t in range(idx - (T_LENGTH - 1), idx):

            fname = frames[t]

            orig_path = os.path.join(
                orig_video_dir,
                fname
            )

            anon_path = os.path.join(
                anon_video_dir,
                Path(fname).stem + ".png"
            )

            if not os.path.exists(anon_path):
                valid = False
                break

            orig_img = Image.open(orig_path).convert("RGB")
            anon_img = Image.open(anon_path).convert("RGB")

            orig_imgs.append(transform(orig_img))
            anon_imgs.append(transform(anon_img))

        if not valid:
            continue

        x_orig = torch.cat(orig_imgs, dim=0).unsqueeze(0)
        x_anon = torch.cat(anon_imgs, dim=0).unsqueeze(0)

        x_orig = x_orig.to(DEVICE)
        x_anon = x_anon.to(DEVICE)

        with torch.no_grad():

            fea_orig, _, _, _ = encoder(x_orig)
            fea_anon, _, _, _ = encoder(x_anon)

        # =================================================
        # KL (AFN-Pose paper style)
        # =================================================

        mu_orig = fea_orig.mean(
            dim=(0, 2, 3)
        )

        var_orig = fea_orig.var(
            dim=(0, 2, 3),
            unbiased=False
        )

        mu_anon = fea_anon.mean(
            dim=(0, 2, 3)
        )

        var_anon = fea_anon.var(
            dim=(0, 2, 3),
            unbiased=False
        )

        kl = (
            torch.log(
                (var_anon + eps)
                /
                (var_orig + eps)
            )
            +
            (
                var_orig
                +
                (mu_orig - mu_anon) ** 2
            )
            /
            (var_anon + eps)
            -
            1
        )

        kl = kl.mean().item()

        # =================================================
        # cosine similarity
        # =================================================

        cos = F.cosine_similarity(
            fea_orig.flatten(),
            fea_anon.flatten(),
            dim=0
        ).item()

        all_kl.append(kl)
        all_cos.append(cos)

# =====================================================
# results
# =====================================================

import numpy as np

all_kl = np.array(all_kl)
all_cos = np.array(all_cos)

print("\n========================")
print("Feature Privacy Analysis")
print("========================")

print(
    f"KL Mean : {all_kl.mean():.4f}"
)

print(
    f"KL Std  : {all_kl.std():.4f}"
)

print(
    f"Cos Mean: {all_cos.mean():.4f}"
)

print(
    f"Cos Std : {all_cos.std():.4f}"
)