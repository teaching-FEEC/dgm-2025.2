import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class adamw_sf(Optimizer):
    r"""Schedule-Free AdamW
    As the name suggests, no scheduler is needed with this optimizer.
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.

    Arguments:
    ---------
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99)).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).

    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 8e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        schedule_free: bool = True,  # noqa: ARG002
        foreach: bool = True,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "r": r,
            "k": 0,
            "warmup_steps": warmup_steps,
            "train_mode": True,
            "weight_sum": 0.0,
            "lr_max": -1.0,
            "weight_lr_power": weight_lr_power,
            "weight_decay": weight_decay,
            "foreach": foreach,
        }
        super().__init__(params, defaults)

    def eval(self) -> None:
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p.data to x
                        p.data.lerp_(end=state["z"], weight=1 - 1 / beta1)
                group["train_mode"] = False

    def train(self) -> None:
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if not train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p.data to y
                        p.data.lerp_(end=state["z"], weight=1 - beta1)
                group["train_mode"] = True

    def step(self, closure: Callable[..., Any] | None = None) -> float:  # type: ignore[reportIncompatibleMethodOverride,override]
        """Performs a single optimization step.

        Arguments:
        ---------
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        """
        loss = closure() if closure is not None else 0.0

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            k = group["k"]
            r = group["r"]
            warmup_steps = group["warmup_steps"]
            weight_lr_power = group["weight_lr_power"]

            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0

            bias_correction2 = 1 - beta2 ** (k + 1)
            lr = group["lr"] * sched * math.sqrt(bias_correction2)

            lr_max = group["lr_max"] = max(lr, group["lr_max"])

            weight = ((k + 1) ** r) * (lr_max**weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight

            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            if not group["train_mode"]:
                msg = "Not in train mode!"
                raise ValueError(msg)

            active_p = [p for p in group["params"] if p.grad is not None]

            for p in active_p:
                if "z" not in self.state[p]:
                    self.state[p]["z"] = torch.clone(p.data)
                    self.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)

            if group["foreach"] and len(active_p) > 0:
                y, grad, exp_avg_sq, z = zip(
                    *[
                        (
                            p.data,
                            p.grad,
                            self.state[p]["exp_avg_sq"],
                            self.state[p]["z"],
                        )
                        for p in active_p
                    ],
                    strict=False,
                )

                # Decay the first and second moment running average coefficient
                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1 - beta2)
                denom = torch._foreach_sqrt(exp_avg_sq)
                torch._foreach_add_(denom, eps)

                # Normalize grad in-place for memory efficiency
                torch._foreach_div_(grad, denom)

                # Weight decay calculated at y
                if decay != 0:
                    torch._foreach_add_(grad, y, alpha=decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad, alpha=lr * (beta1 * (1 - ckp1) - 1))

                # z step
                torch._foreach_sub_(z, grad, alpha=lr)
            else:
                for p in active_p:
                    y = p.data  # Notation to match theory
                    grad = p.grad.data

                    state = self.state[p]

                    z = state["z"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # type: ignore[attr-defined]
                    denom = exp_avg_sq.sqrt().add_(eps)  # type: ignore[attr-defined]

                    # Reuse grad buffer for memory efficiency
                    grad_normalized = grad.div_(denom)  # type: ignore[attr-defined]

                    # Weight decay calculated at y
                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)  # type: ignore[attr-defined]
                    y.add_(grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))  # type: ignore[attr-defined]

                    # z step
                    z.sub_(grad_normalized, alpha=lr)  # type: ignore[attr-defined]

            group["k"] = k + 1
        return loss
