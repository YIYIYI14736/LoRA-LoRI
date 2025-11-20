import torch
import os

state_path = r"F:\SAM-New\LoRA-LoRI\work_dir\b1_4_1折_LoRI\medsam_model_stage2_best.pth"
out_path = os.path.splitext(state_path)[0] + "_params.txt"

state_dict = torch.load(state_path, map_location="cpu")

with open(out_path, "w", encoding="utf-8") as f:
    for key, value in state_dict.items():
        f.write(f"键：{key}, 值：{value}\n")

print("saved:", out_path)