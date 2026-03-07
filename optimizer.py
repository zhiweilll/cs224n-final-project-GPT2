from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups: # param_groups: param groups with different hyperparam (lr/weight decay)
            for p in group["params"]: # p: each param
                if p.grad is None:
                    continue
                grad = p.grad.data # gradient g_t
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                # hyperparameters: learning rate
                alpha = group["lr"]


                ### TODO: Complete the implementation of AdamW here, reading and saving
                ###       your state in the `state` dictionary above.
                ###       The hyperparameters can be read from the `group` dictionary
                ###       (they are lr, betas, eps, weight_decay, as saved in the constructor).
                ###
                ###       To complete this implementation:
                ###       1. Update the first and second moments of the gradients.
                ###       2. Apply bias correction
                ###          (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                ###          also given in the pseudo-code in the project description).
                ###       3. Update parameters (p.data).
                ###       4. Apply weight decay after the main gradient-based updates.
                ###
                ###       Refer to the default project handout for more details.
                
                # Step 0: Initialize state on first step
                if len(state) == 0:
                    state['step'] = 0    # step: t
                    state['exp_avg'] = torch.zeros_like(p.data)     # m_0: 1st moment
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # v_0: 2nd moment
                
                # hyperparams: 
                beta1, beta2 = group['betas'] # Exponential decay rates for the moment estimates
                eps = group['eps'] # epsilon
                weight_decay = group['weight_decay'] # Decoupled Weight Decay Regularization: lamda λ 

                state['step'] += 1
                t = state['step']
                m = state['exp_avg']
                v = state['exp_avg_sq']

                # Step 1: update biased 1st and 2nd moment estimates
                # m_t = β₁·m_{t-1} + (1-β₁)·g_t
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # v_t = β₂·v_{t-1} + (1-β₂)·g_t²
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Step 2: efficient bias-corrected learning rate
                # α_t = α · √(1-β₂ᵗ) / (1-β₁ᵗ)
                alpha_t = alpha * math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t) 

                # Step 3: parameter update  θ_t = θ_{t-1} - α_t · m_t / (√v_t + ε)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # Step 4: decoupled weight decay  θ_t = θ_t - α · λ · θ_t
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-alpha * weight_decay)



        return loss
