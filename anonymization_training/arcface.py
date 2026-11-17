import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# ArcFace model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

def get_video_id(path):
    return path.replace("\\", "/").split("/")[-2]

def extract_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    faces = app.get(img)

    if len(faces) == 0:
        return None

    # 取最大脸（论文默认策略）
    face = max(faces, key=lambda x: x.det_score)
    return face.embedding

def extract_video_embedding(video_folder):
    embs = []

    frames = sorted(os.listdir(video_folder))
    frames = frames[::5]

    for f in frames:
        path = os.path.join(video_folder, f)
        emb = extract_face_embedding(path)

        if emb is not None:
            embs.append(emb)

    if len(embs) == 0:
        return None

    # mean pooling = video identity representation
    return np.mean(embs, axis=0)

def build_dataset(root_dir):
    features = {}
    
    for video in sorted(os.listdir(root_dir)):
        video_path = os.path.join(root_dir, video)

        if not os.path.isdir(video_path):
            continue

        emb = extract_video_embedding(video_path)

        if emb is not None:
            vid = video  # ⚠️ 强制用 folder name
            features[vid] = emb

    return features

def retrieval_attack(orig_feats, anon_feats):

    # ✅ 1. 只取共同存在的 video
    common_keys = sorted(set(orig_feats.keys()) & set(anon_feats.keys()))

    print("Matched videos:", len(common_keys))

    # ❗ 如果太少，提示问题
    if len(common_keys) == 0:
        raise ValueError("No matched identities between original and anonymized!")

    # ✅ 2. 构建对齐数据
    orig = np.array([orig_feats[k] for k in common_keys])
    anon = np.array([anon_feats[k] for k in common_keys])

    # 3. cosine similarity
    sim = cosine_similarity(anon, orig)

    # 4. retrieval accuracy
    top1 = 0
    for i in range(len(common_keys)):
        if np.argmax(sim[i]) == i:
            top1 += 1

    return top1 / len(common_keys)

def cosine_identity_leakage(orig_feats, anon_feats):

    # ✔ 只取共同 identity
    common_keys = sorted(set(orig_feats.keys()) & set(anon_feats.keys()))

    print("Matched identities:", len(common_keys))

    if len(common_keys) == 0:
        raise ValueError("No matched identities!")

    orig = np.array([orig_feats[k] for k in common_keys])
    anon = np.array([anon_feats[k] for k in common_keys])

    # cosine similarity matrix
    sim = cosine_similarity(orig, anon)

    # leakage score（越高越危险）
    diag = np.diag(sim)
    return np.mean(diag)

def rank_k_accuracy(original_feats, anon_feats, k=5):

    common = sorted(set(original_feats) & set(anon_feats))

    orig = np.array([original_feats[i] for i in common])
    anon = np.array([anon_feats[i] for i in common])

    sim_matrix = cosine_similarity(anon, orig)

    correct = 0

    for i in range(len(common)):
        topk = np.argsort(sim_matrix[i])[::-1][:k]
        if i in topk:
            correct += 1

    return correct / len(common)

if __name__ == "__main__":

    # ===== Avenue =====
    orig_dir = "/mnt/private/xwy/aed-mae/data/avenue/test/frames"
    anon_dir = "/mnt/private/xwy/SPAct/anonymization_training/ppopps_av2"

    print("Extracting original features...")
    orig_feats = build_dataset(orig_dir)

    print("Extracting anonymized features...")
    anon_feats = build_dataset(anon_dir)
    print("ORIG sample:", list(orig_feats.keys())[:10])
    print("ANON sample:", list(anon_feats.keys())[:10])

    # ===== evaluation =====
    top1 = retrieval_attack(orig_feats, anon_feats)
    cos_leak = cosine_identity_leakage(orig_feats, anon_feats)
    rank5 = rank_k_accuracy(orig_feats, anon_feats, k=5)

    print("\n===== Privacy Evaluation (ArcFace Attack) =====")
    print(f"Top-1 Retrieval Accuracy: {top1:.4f}")
    print(f"Mean Cosine Leakage: {cos_leak:.4f}")
    print(f"Rank-5 Accuracy: {rank5:.4f}")