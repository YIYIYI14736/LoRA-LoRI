"""
2025-11-12
LoRI结构实现
即冻结随机初始化权重A，通过构建稀疏矩阵，确保权重矩阵B仅一部分参数被更新
其中稀疏掩码为0的部分，采用hook机制，直接将部分梯度清0
"""

import torch
import torch.nn as nn
import numpy as np

from src.segment_anything import build_sam_vit_b
from src.segment_anything.modeling.sam import Sam

from safetensors import safe_open
from safetensors.torch import save_file

from .lora import LoRA_sam, LoRA_qkv

class LoRI_sam(LoRA_sam):
    def __init__(self, sam_model: Sam, rank: int, lora_layer=None):
        """
        冻结图像编码器的原始参数，在指定的block上加入lori，随机初始化权重举证A，B为0
        """
        super(LoRI_sam, self).__init__(sam_model=sam_model, rank=rank, lora_layer=lora_layer)

        for w_A in self.A_weights:
            for p in w_A.parameters():
                p.requires_grad = False

        for w_B in self.B_weights:
            for p in w_B.parameters():
                p.requires_grad = True

        self.lori_masks = None

    def reset_parameters(self):
        """
        权重初始化
        """
        print("=== 采用随机高斯 初始化 LoRI 权重 ===")
        for w_A in self.A_weights: # r * d
            std = 1 / np.sqrt(w_A.weight.shape[1])
            nn.init.normal_(w_A.weight, mean=0.0, std=std)
        for w_B in self.B_weights: # d * r
            nn.init.zeros_(w_B.weight)

    def get_B_parameters(self):
        """
        返回所有B权重矩阵的参数，便于构建优化器以及统计
        """
        return [w_B.weight for w_B in self.B_weights]

    def get_B_all_vector(self):
        """
        将所有B权重矩阵的参数，展平成一个向量
        """
        with torch.no_grad():
            flat_list = [w_B.weight.detach().view(-1) for w_B in self.B_weights]
            if len(flat_list) == 0:
                return None
            return torch.cat(flat_list, dim=0)

    def compute_global_top_masks(self, sparsity_ratio: float = 0.9, device: str = "cuda"):
        """
        基于全部B的参数，构建稀疏矩阵
        输出列表，存放所有B矩阵对应的mask部分
        """
        all_B = self.get_B_all_vector()
        assert all_B is not None,  "No B weights found"

        # k为保留的权重的个数
        total_num = all_B.numel()
        k = int((1.0 - sparsity_ratio) * total_num)
        k = max(1, k)

        # 取绝对值后，top_k取取出阈值，小于阈值的位置为0
        abs_vals = all_B.abs()
        if k >= total_num:
            threshold = abs_vals.min() - 1.0 # 所有位置都大于阈值
        else:
            threshold = torch.topk(abs_vals, k).values[-1]
            # values, _ = torch.topk(abs_vals, k, largest=None, sorted=True)
            # threshold = values[-1]

        masks = []
        for w_B in self.B_weights:
            w = w_B.weight.detach()
            mask = (w.abs() >= threshold).to(device)
            masks.append(mask)

        self.lori_masks = masks
        return masks, threshold.item()

    def apply_mask_to_B(self, zero_out_unselected: bool = True):
        """
        在第二阶段训练开始前，把所有B的非mask部分，直接设为0
        """
        assert self.lori_masks is not None, "You must call compute_global_topk_masks() before apply_mask_to_B()."
        
        with torch.no_grad():
            for w_B, mask in zip(self.B_weights, self.lori_masks):
                mask = mask.to(w_B.weight.device)
                if zero_out_unselected:
                    w_B.weight.data *= mask

    def register_gradient_masks(self):
        """
        为所有B权重矩阵，注册一个hook，在反向传播时，根据mask，将未选中的位置的梯度设为0
        grad = grad * mask
        """
        assert self.lori_masks is not None, "You must call compute_global_topk_masks() before register_gradient_masks()."
        
        for w_B, mask in zip(self.B_weights, self.lori_masks):
            weight = w_B.weight

            def _hook(grad, m=mask):
                m = m.to(grad.device)
                return grad * m
            
            weight.register_hook(_hook)

    def save_lori_parameters(self, filename: str):
        """
        保存 LoRI 的 A/B 权重和 (可选) mask。
        这里我们仍然沿用原来的 A/B 命名，另加 lori_mask_xxx。
        """
        num_layer = len(self.A_weights)

        a_tensor = {f"w_a_{i:03d}": self.A_weights[i].weight for i in range(num_layer)}
        b_tensor = {f"w_b_{i:03d}": self.B_weights[i].weight for i in range(num_layer)}

        merged = {**a_tensor, **b_tensor}

        if self.lori_masks is not None:
            for i, mask in enumerate(self.lori_masks):
                merged[f"mask_{i:03d}"] = mask.to(torch.bool)

        save_file(merged, filename)

    def load_lori_parameters(self, filename: str, load_mask: bool = True):
        """
        从 safetensors 中恢复 A/B 权重和（可选）mask。
        """
        with safe_open(filename, framework="pt") as f:
            for i, w_A in enumerate(self.A_weights):
                key = f"w_a_{i:03d}"
                if key in f.keys():
                    w_A.weight = nn.Parameter(f.get_tensor(key))

            for i, w_B in enumerate(self.B_weights):
                key = f"w_b_{i:03d}"
                if key in f.keys():
                    w_B.weight = nn.Parameter(f.get_tensor(key))

            if load_mask:
                masks = []
                for i in range(len(self.B_weights)):
                    key = f"mask_{i:03d}"
                    if key in f.keys():
                        masks.append(f.get_tensor(key).to(torch.bool))
                if len(masks) == len(self.B_weights):
                    self.lori_masks = masks

