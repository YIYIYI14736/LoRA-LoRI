import sys
import os
import torch
import matplotlib.pyplot as plt
from safetensors import safe_open
from safetensors.torch import load_file
import numpy as np

num = 4
rank = 4
batch = 1
DEFAULT_SAFE_PATH = rf"F:\SAM-New\LoRA-LoRI\work_dir\b{num}_{rank}_{batch}折_LoRI\sam_lori_stage1_dense.safetensors"
DEFAULT_OUT_DIR = rf"F:\SAM-New\LoRA-LoRI\spence"

def visualize_lori_masks_qv(safetensors_path, out_dir):
    mask_arrays = []
    mask_names = []
    b_shapes = {}
    with safe_open(safetensors_path, framework="pt") as f:
        keys = list(f.keys())
        mask_keys = sorted([k for k in keys if k.startswith("mask_")])
        if len(mask_keys) == 0:
            raise RuntimeError("no mask_* keys found in safetensors")
        for k in mask_keys:
            m = f.get_tensor(k).to(torch.bool).float().cpu().numpy()
            mask_arrays.append(m)
            mask_names.append(k)
            kb = "w_b_" + k.split("_")[-1]
            if kb in keys:
                bs = tuple(f.get_tensor(kb).shape)
                b_shapes[k] = bs
            else:
                b_shapes[k] = tuple(m.shape)
    q_arrays = mask_arrays[0::2]
    v_arrays = mask_arrays[1::2]
    q_names = mask_names[0::2]
    v_names = mask_names[1::2]
    cols = 4
    rows_q = int(np.ceil(len(q_arrays) / cols)) if len(q_arrays) > 0 else 1
    rows_v = int(np.ceil(len(v_arrays) / cols)) if len(v_arrays) > 0 else 1
    os.makedirs(out_dir, exist_ok=True)
    fig_q, axes_q = plt.subplots(rows_q, cols, figsize=(4 * cols, 3 * rows_q))
    for idx in range(rows_q * cols):
        r = idx // cols
        c = idx % cols
        ax = axes_q[r][c] if rows_q > 1 else axes_q[c]
        if idx < len(q_arrays):
            arr = q_arrays[idx].T
            ax.imshow(arr, cmap="binary", aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            bs = b_shapes[q_names[idx]]
            blk = idx
            ax.set_title(f"Block {blk} Q  {q_names[idx]} → w_b_{q_names[idx].split('_')[-1]}  {arr.shape[0]}x{arr.shape[1]}")
        else:
            ax.axis("off")
    plt.tight_layout()
    out_q = os.path.join(out_dir, f"q_masks_b{num}_{rank}_{batch}.png")
    plt.savefig(out_q, dpi=150)
    plt.close()
    fig_v, axes_v = plt.subplots(rows_v, cols, figsize=(4 * cols, 3 * rows_v))
    for idx in range(rows_v * cols):
        r = idx // cols
        c = idx % cols
        ax = axes_v[r][c] if rows_v > 1 else axes_v[c]
        if idx < len(v_arrays):
            arr = v_arrays[idx].T
            ax.imshow(arr, cmap="binary", aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            bs = b_shapes[v_names[idx]]
            blk = idx
            ax.set_title(f"Block {blk} V  {v_names[idx]} → w_b_{v_names[idx].split('_')[-1]}  {arr.shape[0]}x{arr.shape[1]}")
        else:
            ax.axis("off")
    plt.tight_layout()
    out_v = os.path.join(out_dir, f"v_masks_b{num}_{rank}_{batch}.png")
    plt.savefig(out_v, dpi=150)
    plt.close()
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        in_path = DEFAULT_SAFE_PATH
        out_dir = DEFAULT_OUT_DIR
    else:
        in_path = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_OUT_DIR
    if not os.path.isfile(in_path):
        print("file not found:", in_path)
        sys.exit(1)
    visualize_lori_masks_qv(in_path, out_dir)
    print("saved:", os.path.join(out_dir, f"q_masks_b{num}_{rank}_{batch}.png"))
    print("saved:", os.path.join(out_dir, f"v_masks_b{num}_{rank}_{batch}.png"))