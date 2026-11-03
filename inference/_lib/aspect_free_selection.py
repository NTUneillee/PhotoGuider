"""Canonical DPP K10 selection machinery.

Greedy DPP + quality construction used by infer.py's K10 selection (the quality model is
inference/models/dpp_quality.pkl). Function bodies are byte-identical extracts of the validated
benchmark recipe."""
import numpy as np


def pairwise_iou_vec(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    xx1 = np.maximum(x1[:, None], x1[None, :])
    yy1 = np.maximum(y1[:, None], y1[None, :])
    xx2 = np.minimum(x2[:, None], x2[None, :])
    yy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def dpp_greedy(q, sim, K):
    N = q.shape[0]
    Km = min(K, N)
    if Km == 0:
        return np.array([], dtype=int)
    L = np.outer(q, q) * sim
    d_sq = np.copy(np.diag(L))
    C = np.zeros((N, Km), dtype=np.float32)
    selected = []
    for it in range(Km):
        i = int(np.argmax(d_sq))
        if d_sq[i] <= 1e-12:
            break
        selected.append(i)
        if it == Km - 1:
            break
        e = (L[:, i] - C[:, :it] @ C[i, :it]) / np.sqrt(d_sq[i])
        C[:, it] = e
        d_sq = d_sq - e ** 2
        d_sq[i] = -np.inf
    return np.array(selected)


def run_dpp_per_image(df_pool, q_array, K=10, beta=4.0, src_penalty=0.05):
    """Greedy DPP K per photo_id. pid keys kept as-is (str for FLMS)."""
    selections = {}
    for pid, g in df_pool.groupby("photo_id"):
        n = len(g)
        if n == 0:
            continue
        g_idx = g.index.values
        q = q_array[g_idx].astype(np.float32)
        boxes = g[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
        W = float(g["img_w"].iloc[0])
        H = float(g["img_h"].iloc[0])
        boxes_n = boxes / np.array([W, H, W, H], dtype=np.float32)
        iou_mat = pairwise_iou_vec(boxes_n)
        np.fill_diagonal(iou_mat, 1.0)
        sim = np.exp(-beta * (1.0 - iou_mat))
        if src_penalty > 0.0:
            srcs = np.asarray(g["src"].astype(str).values)
            same_src = (srcs[:, None] == srcs[None, :]).astype(np.float32)
            sim = sim + src_penalty * same_src
            np.fill_diagonal(sim, 1.0)
        sel = dpp_greedy(q, sim, K)
        if len(sel) > 0:
            selections[pid] = g_idx[sel].tolist()
    return selections


def make_q_dpp(df_pool, q_lgbm, vmix_boost=0.5, aspect_alpha=1.5):
    aspect = np.maximum(0.0, 1.0 - aspect_alpha * df_pool["log_aspect_diff"].values)
    return (q_lgbm + vmix_boost * df_pool["vmix_max"].values) * aspect
