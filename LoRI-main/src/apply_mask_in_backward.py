from typing import Any, Dict, Iterable, List, no_type_check, Type

import torch

__all__: List[str] = []

# We are doing this since FSDP does not support `register_post_accumulate_grad_hook` yet.

param_to_optim_hook_handle_map = torch.utils.weak.WeakTensorKeyDictionary()
param_to_acc_grad_map = torch.utils.weak.WeakTensorKeyDictionary()

class MaskedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, mask=None, grad_norm_strategy='even', lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, lr_scheduler_cls=None, lr_scheduler_kwargs=None, max_grad_norm=None):
        super(MaskedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        self.mask = mask.bool() if mask is not None else None
        self.max_grad_norm = max_grad_norm
        # Initialize the learning rate scheduler, if provided
        self.lr_scheduler = None
        if lr_scheduler_cls is not None:
            if not issubclass(lr_scheduler_cls, torch.optim.lr_scheduler.LRScheduler):
                raise ValueError("lr_scheduler_cls must be a subclass of torch.optim.lr_scheduler._LRScheduler")
            self.lr_scheduler = lr_scheduler_cls(self, **(lr_scheduler_kwargs or {}))

    def step(self, closure=None):
        """Performs a single optimization step with gradient masking."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Apply mask to gradients before the optimization step
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if self.mask is not None:
                            if self.mask.shape == p.grad.data.shape:
                                p.grad.data[~self.mask] = 0
                            else:
                                raise ValueError(f"Mask shape {self.mask.shape} does not match gradient shape {p.grad.data.shape}")
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(p, self.max_grad_norm)
        
        # Call the original RMSprop step function
        super(MaskedRMSprop, self).step(closure)

        # Step the learning rate scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss


class MaskedAdamW(torch.optim.AdamW):
    def __init__(self, params, mask=None, grad_norm_strategy='even', lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, lr_scheduler_cls=None, lr_scheduler_kwargs=None, max_grad_norm=None):
        super(MaskedAdamW, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.mask = mask.bool() if mask is not None else None
        self.max_grad_norm = max_grad_norm

        # Initialize the learning rate scheduler, if provided
        self.lr_scheduler = None
        if lr_scheduler_cls is not None:
            if not issubclass(lr_scheduler_cls, torch.optim.lr_scheduler.LRScheduler):
                raise ValueError("lr_scheduler_cls must be a subclass of torch.optim.lr_scheduler._LRScheduler")
            self.lr_scheduler = lr_scheduler_cls(self, **(lr_scheduler_kwargs or {}))

    def step(self, closure=None):
        """Performs a single optimization step with gradient masking and optional gradient clipping."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Apply mask to gradients before the optimization step
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if self.mask is not None:
                            if self.mask.shape == p.grad.data.shape:
                                p.grad.data[~self.mask] = 0
                            else:
                                raise ValueError(f"Mask shape {self.mask.shape} does not match gradient shape {p.grad.data.shape}")
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(p, self.max_grad_norm)

        # Call the original AdamW step function
        super(MaskedAdamW, self).step(closure)

        # Step the learning rate scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss
    

@no_type_check
def _apply_masked_optimizer_in_backward(optimizer_class, named_params, mask_dict, optimizer_kwargs, use_mask: bool = False, register_hook: bool = True):
    
    @no_type_check
    def _apply_masked_optimizer_in_backward_to_param(param: torch.nn.Parameter, mask, optimizer_kwargs_inner) -> None:
        if not param.requires_grad:
            return
        # view_as creates a node in autograd graph that allows us access to the
        # parameter's AccumulateGrad autograd function object. We register a
        # hook on this object to fire the optimizer when the gradient for
        # this parameter is ready (has been accumulated into .grad field)

        # Don't create a new acc_grad if we already have one
        # i.e. for shared parameters or attaching multiple optimizers to a param.
        if param not in param_to_acc_grad_map:
            param_to_acc_grad_map[param] = param.view_as(param).grad_fn.next_functions[0][0]
            
        if optimizer_class == torch.optim.AdamW:
            masked_optimizer_class = MaskedAdamW
        elif optimizer_class == torch.optim.RMSprop:
            masked_optimizer_class = MaskedRMSprop
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_class}")
        
        optimizer = masked_optimizer_class([param], mask, **optimizer_kwargs_inner)

        if not hasattr(param, "_in_backward_optimizers"):
            param._in_backward_optimizers = []  # type: ignore[attr-defined]
            param._optimizer_classes = []  # type: ignore[attr-defined]
            param._optimizer_kwargs = []  # type: ignore[attr-defined]

        param._in_backward_optimizers.append(optimizer)  # type: ignore[attr-defined]
        param._optimizer_classes.append(optimizer_class)  # type: ignore[attr-defined]
        param._optimizer_kwargs.append(optimizer_kwargs_inner)  # type: ignore[attr-defined]

        if not register_hook:
            return

        def optimizer_hook(*_unused) -> None:
            for opt in param._in_backward_optimizers:  # type: ignore[attr-defined]
                opt.step()
            param.grad = None

        handle = param_to_acc_grad_map[param].register_hook(optimizer_hook)  # type: ignore[attr-defined]
        if param not in param_to_optim_hook_handle_map:
            param_to_optim_hook_handle_map[param] = []
        param_to_optim_hook_handle_map[param].append(handle)



    if optimizer_kwargs['grad_norm_strategy'] == 'even':
        max_grad_norm = optimizer_kwargs['max_grad_norm']
    elif optimizer_kwargs['grad_norm_strategy'] == 'proportional':
        num_params = len(list(named_params))
        max_grad_norm = optimizer_kwargs['max_grad_norm'] / (num_params ** 0.5)
    else:
        raise ValueError(f"Invalid grad_norm_strategy: {optimizer_kwargs['grad_norm_strategy']}")
    
    optimizer_kwargs['max_grad_norm'] = max_grad_norm
    
    for name, param in named_params:
        optimizer_kwargs_inner = optimizer_kwargs.copy()
        if "lora_B.default.weight" in name and use_mask:
            mask_name = name.replace("_checkpoint_wrapped_module.", "").replace("_fsdp_wrapped_module.", "").replace("lora_B.default.weight", "lora_B.mask")
            mask = mask_dict[mask_name].to(param.device).detach()
            _apply_masked_optimizer_in_backward_to_param(param=param,
                                                         mask=mask,
                                                         optimizer_kwargs_inner=optimizer_kwargs_inner)
        else:
            _apply_masked_optimizer_in_backward_to_param(param=param,
                                                         mask=None,
                                                         optimizer_kwargs_inner=optimizer_kwargs_inner)


def _get_in_backward_optimizers(module: torch.nn.Module) -> List[torch.optim.Optimizer]:
    """
    Return a list of in-backward optimizers applied to ``module``'s parameters. Note that these
    optimizers are not intended to directly have their ``step`` or ``zero_grad`` methods called
    by the user and are intended to be used for things like checkpointing.

    Args:
        module: (torch.nn.Module): model to retrieve in-backward optimizers for

    Returns:
        List[torch.optim.Optimizer]: the in-backward optimizers.

    Example::
        _apply_optimizer_in_backward(torch.optim.SGD, model.parameters(), {'lr': 0.01})
        optims = _get_optimizers_in_backward(model)
    """
    optims: List[torch.optim.Optimizer] = []
    for param in module.parameters():
        optims.extend(getattr(param, "_in_backward_optimizers", []))

    return optims
