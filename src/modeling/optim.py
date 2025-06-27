"""
Advanced optimization utilities for NER model training with:
- Custom learning rate schedulers
- Gradient optimization techniques
- Memory-efficient optimizers
- Automated hyperparameter tuning
"""

import math
import torch
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable, Iterable, Tuple, Optional
from collections import defaultdict
import numpy as np
from transformers import get_scheduler
from ray import tune
import logging

logger = logging.getLogger(__name__)

# --------------------------
# 1. Advanced Optimizer Classes
# --------------------------

class FusedAdamW(AdamW):
    """NVIDIA's fused AdamW implementation for 30% faster training"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, fused=True):
        try:
            from apex.optimizers import FusedAdam
            super().__init__(params, lr=lr, betas=betas, eps=eps, 
                           weight_decay=weight_decay, amsgrad=amsgrad)
            self.fused_impl = FusedAdam(params, lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, amsgrad=amsgrad)
            self.fused = fused
        except ImportError:
            logger.warning("Apex not installed, falling back to vanilla AdamW")
            self.fused = False
            super().__init__(params, lr=lr, betas=betas, eps=eps,
                           weight_decay=weight_decay, amsgrad=amsgrad)

    def step(self, closure=None):
        if self.fused:
            return self.fused_impl.step(closure)
        return super().step(closure)

class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (Sharpness Aware Minimization)
    https://arxiv.org/abs/2010.01412
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                if group["adaptive"]:
                    e_w *= torch.norm(p) / (torch.norm(p.grad) + 1e-6)
                p.add_(e_w)  # climb to the local maximum
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to the minima
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                torch.norm(p.grad, 2.0) 
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), 
            2.0
        )
        return norm

# --------------------------
# 2. Learning Rate Schedulers
# --------------------------

class TriangularCLR(LambdaLR):
    """Cyclical Learning Rates for better convergence"""
    def __init__(self, optimizer, base_lr, max_lr, step_size, mode='triangular', 
                 gamma=1.0, scale_fn=None, scale_mode='cycle', last_epoch=-1):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.0**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, epoch):
        cycle = math.floor(1 + epoch/(2 * self.step_size))
        x = abs(epoch/self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * self.scale_fn(epoch)
        return lr / self.base_lr

class WarmupCosineWithHardRestarts(LambdaLR):
    """Cosine with warmup and hard restarts (SGDR)"""
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, 
                 num_cycles=1.0, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(self.num_cycles) * progress) % 1.0)))

# --------------------------
# 3. Gradient Optimization
# --------------------------

class GradientAccumulator:
    """Memory-efficient gradient accumulation"""
    def __init__(self):
        self._grads = defaultdict(lambda: None)
        self._num_accumulations = 0

    def accumulate(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if self._grads[name] is None:
                    self._grads[name] = torch.zeros_like(param.grad)
                self._grads[name].add_(param.grad)
        self._num_accumulations += 1

    def apply(self, optimizer, zero_grad=True):
        for name, param in model.named_parameters():
            if self._grads[name] is not None:
                param.grad = self._grads[name] / self._num_accumulations
        optimizer.step()
        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        self._grads = defaultdict(lambda: None)
        self._num_accumulations = 0

def clip_grad_norm_custom(parameters, max_norm, norm_type=2, error_if_nonfinite=True):
    """Enhanced gradient clipping with error tracking"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    if norm_type == math.inf:
        norms = [p.grad.detach().abs().max() for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), 
            norm_type
        )
    
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(f"Grad norm is {total_norm}")
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    
    return total_norm

# --------------------------
# 4. Hyperparameter Optimization
# --------------------------

def ray_tune_hyperparams(config: dict, train_func: Callable) -> dict:
    """Automated hyperparameter tuning with Ray Tune"""
    analysis = tune.run(
        train_func,
        config=config,
        resources_per_trial={"cpu": 8, "gpu": 1},
        metric="f1_score",
        mode="max",
        num_samples=50,
        scheduler=tune.schedulers.ASHAScheduler(
            metric="f1_score",
            mode="max",
            max_t=100,
            grace_period=10,
            reduction_factor=2
        ),
        progress_reporter=tune.CLIReporter(
            parameter_columns=list(config.keys()),
            metric_columns=["f1_score", "training_iteration"]
        )
    )
    return analysis.best_config

def find_optimal_lr(model, train_loader, optimizer_class=AdamW):
    """LR range test following Leslie Smith's method"""
    lrs = np.logspace(-7, 1, 100)
    losses = []
    
    for lr in lrs:
        optimizer = optimizer_class(model.parameters(), lr=lr)
        model.train()
        batch = next(iter(train_loader))
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    
    # Find point of steepest descent
    gradients = np.gradient(losses)
    optimal_idx = np.argmin(gradients)
    return lrs[optimal_idx]

# --------------------------
# 5. Utility Functions
# --------------------------

def get_optimizer_and_scheduler(
    model, 
    train_steps: int, 
    optimizer_type: str = "adamw",
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    scheduler_type: str = "linear",
    use_sam: bool = False
) -> Tuple[Optimizer, LambdaLR]:
    """Factory method for optimizer/scheduler setup"""
    
    # Set up optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizers = {
        "adamw": AdamW,
        "fused_adamw": FusedAdamW,
        "sam": SAM
    }
    
    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    if optimizer_type == "sam":
        optimizer = SAM(optimizer_grouped_parameters, base_optimizer=AdamW, rho=0.05)
    else:
        optimizer = optimizers[optimizer_type](
            optimizer_grouped_parameters, 
            lr=learning_rate
        )
    
    # Set up scheduler
    num_warmup_steps = int(train_steps * warmup_ratio)
    
    schedulers = {
        "linear": get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=train_steps
        ),
        "cosine": WarmupCosineWithHardRestarts(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=train_steps
        ),
        "triangular": TriangularCLR(
            optimizer,
            base_lr=learning_rate/10,
            max_lr=learning_rate*10,
            step_size=train_steps//4
        )
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return optimizer, schedulers[scheduler_type]

def log_optimizer_stats(optimizer: Optimizer, logger: logging.Logger):
    """Log detailed optimizer statistics"""
    stats = {
        "lr": optimizer.param_groups[0]['lr'],
        "momentum": optimizer.param_groups[0].get('betas', (0,0))[0],
        "weight_decay": optimizer.param_groups[0]['weight_decay']
    }
    logger.info(f"Optimizer Stats: {stats}")