import torch
from torch.optim import Adam

class MaskedAdam(Adam):
    
    def __init__(self, params, masks, lr=1e-3, weight_decay=0.0, **kwargs):
        super().__init__(params, lr=lr, weight_decay=weight_decay, **kwargs)

        self.param_to_mask = {}
        for p, m in zip(params, masks):
            self.param_to_desk[id(p)] = m.bool()

    @torch.no_grad()
    def step(self,closure=None):
        for group in self.param_to_mask:
            for p in group["params"]:
                if p.grad is None:
                    continue
                mask = self.param_to_mask.get(id(p), None)
                if mask is not None:
                    p.grad *= mask.to(p.grad.device)

        return super().step(closure)
