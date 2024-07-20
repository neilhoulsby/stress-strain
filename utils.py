import torch
import numpy as np


def create_learning_rate_scheduler(warmup_steps=1000, total_steps=10000):
    def lr_lambda(step):
        lr = 1.0
        lr *= min(1.0, step / warmup_steps)
        lr *= min(1.0, (total_steps - step) / (total_steps - warmup_steps))
        return lr

    return lr_lambda


def compute_l2(predictions, targets, padding=None):
    if predictions.ndim != targets.ndim:
        raise ValueError(
            f"Incorrect shapes. Got shape {predictions.shape} predictions and {targets.shape} targets"
        )
    padding = padding or torch.zeros(
        predictions.shape[:-1], dtype=torch.bool, device=predictions.device
    )

    predictions = predictions * ~padding.unsqueeze(-1)
    targets = targets * ~padding.unsqueeze(-1)
    loss = ((predictions - targets) ** 2).sum(dim=-1)
    return loss.mean()


def compute_hinge(values):
    assert values.dim() == 2, f"{values.dim()} != 2"
    loss = torch.clamp(values, min=0)
    return loss.mean()


def compute_losses(
    py,
    pdy,
    physics_aux,
    y,
    dy,
    padding=None,
    deltas_loss_weight=0.0,
    physics_loss_weight=0.0,
):
    l = compute_l2(py, y)
    ld = compute_l2(pdy, dy)
    l2_loss = (1 - deltas_loss_weight) * l + deltas_loss_weight * ld
    if physics_aux is not None:
        physics_loss = compute_hinge(physics_aux)
    else:
        physics_loss = torch.zeros([])
        assert physics_loss_weight == 0.0

    loss = l2_loss + physics_loss_weight * physics_loss
    return {
        "loss": loss,
        "l2_loss": l2_loss,
        "physics_loss": physics_loss,
    }


def forward(model, inputs, deltas_loss_weight, physics_loss_weight):
    py, pdy, physics_aux = model(inputs)
    y = inputs["y"]
    dy = build_deltas(y)
    return compute_losses(
        py=py,
        pdy=pdy,
        physics_aux=physics_aux,
        y=y,
        dy=dy,
        deltas_loss_weight=deltas_loss_weight,
        physics_loss_weight=physics_loss_weight,
    )


def build_deltas(x):
    dx = x[:, 1:, :] - x[:, :-1, :]
    first_dx = torch.zeros((x.shape[0], 1, x.shape[2]), dtype=x.dtype, device=x.device)
    dx = torch.cat([first_dx, dx], dim=1)
    return dx
