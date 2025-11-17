"""
2025-11-17
MaskedAdam 是 Adam 优化器的一个变体，它在更新参数时考虑了一个掩码（mask）。
"""
import torch
from torch.optim import Adam

class MaskedAdam(Adam):
    
    def __init__(self, params, mask=None, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False):
        
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        
        if isinstance(mask, (list, tuple)):
            self.param_to_mask = {}
            all_prarmals = []

            for group in self.param_groups:
                for p in group["params"]:
                    all_prarmals.append(p)

            if len(all_prarmals) != len(mask):
                raise ValueError("The number of parameters and masks must be the same.")

            for p, m in zip(all_prarmals, mask):
                if m is None:
                    continue
                self.param_to_mask[id(p)] = m.bool()

            self.global_mask = None

        else:
            self.param_to_mask = {}
            if mask is not None and not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            self.global_mask = mask.bool() if mask is not None else None

    
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                m = self.param_to_mask.get(id(p), None)
                if m is not None:
                    if m.shape != g.shape:
                        raise ValueError(f"mask 形状 {m.shape} 与 grad 形状 {g.shape} 不一致")
                    g[~m.to(g.device)] = 0.0
                
                elif self.global_mask is not None:
                    if self.global_mask.shape != g.shape:
                        raise ValueError(f"global mask 形状 {self.global_mask.shape} 与 grad 形状 {g.shape} 不一致")
                    g[~self.global_mask.to(g.device)] = 0.0
        
        super().step(closure)

        return loss
                    



