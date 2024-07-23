import math
import os
import torch
import torch.optim as optim
import numpy as np
import time
from data import get_datasets
from model import Transformer, TransformerConfig
from utils import forward, create_learning_rate_scheduler


def train(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.manual_seed(hparams["random_seed"])
    torch.set_float32_matmul_precision("high")
    model_path = (
        os.path.join(hparams["model_dir"], "model.pth")
        if hparams["model_dir"]
        else None
    )

    train_loader, eval_loader = get_datasets(
        batch_size=hparams["batch_size"], train_size=hparams["train_size"]
    )

    config = TransformerConfig(
        max_len=hparams["max_len"],
        num_layers=hparams["num_layers"],
        hidden_dim=hparams["hidden_dim"],
        mlp_dim=hparams["mlp_dim"],
        num_heads=hparams["num_heads"],
        dropout_rate=hparams["dropout_rate"],
        attention_dropout_rate=hparams["attention_dropout_rate"],
        causal_x=hparams["causal_x"],
        physics_decoder=hparams["physics_decoder"],
    )
    total_steps = math.ceil(hparams["total_examples"] / hparams["batch_size"])
    model = Transformer(config).to(device)
    model = torch.compile(model, backend="inductor")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        create_learning_rate_scheduler(
            warmup_steps=hparams["warmup_steps"], total_steps=total_steps
        ),
    )

    print(f"Training started, running for {total_steps} steps")
    model.train()
    train_metrics = []
    all_metrics = []
    step = 0
    last_eval_step = 0
    best_eval_loss = float("inf")
    tick = time.time()
    while step < total_steps:
        for batch in train_loader:
            if step % hparams["eval_freq"] == 0 or step == total_steps - 1:
                tock = time.time()
                train_summary = {"learning_rate": scheduler.get_last_lr()[0]}
                sequences_per_sec = (
                    (hparams["batch_size"] * (step - last_eval_step) / (tock - tick))
                    if step > 0
                    else -1
                )
                if step > 0:
                    train_summary.update(
                        {
                            k: np.mean([m[k] for m in train_metrics])
                            for k in train_metrics[0]
                        }
                    )
                    train_metrics = []

                model.eval()
                eval_metrics = []
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                        metrics = forward(
                            model,
                            eval_batch,
                            hparams["deltas_loss_weight"],
                            hparams["physics_loss_weight"],
                        )
                        eval_metrics.append(
                            {
                                k: v.detach().item() if v is not None else None
                                for k, v in metrics.items()
                            }
                        )

                eval_summary = {
                    k: np.mean([m[k] for m in eval_metrics]) for k in eval_metrics[0]
                }

                metrics_summary = {
                    "step": step,
                    "train_loss": train_summary.get("loss", -1),
                    "train_l2_loss": train_summary.get("l2_loss", -1),
                    "train_physics_loss": train_summary.get("physics_loss", -1),
                    "eval_loss": eval_summary["loss"],
                    "eval_l2_loss": eval_summary["l2_loss"],
                    "eval_physics_loss": eval_summary["physics_loss"],
                    "sequences_per_sec": sequences_per_sec,
                }
                all_metrics.append(metrics_summary)

                print(
                    f"Step: {step:04d},\ttrain loss {metrics_summary['train_loss']:.4f},\t"
                    f"train l2 {metrics_summary['train_l2_loss']:.4f},\ttrain aux {metrics_summary['train_physics_loss']:.4f},\t"
                    f"eval loss {metrics_summary['eval_loss']:.4f},\teval l2 {metrics_summary['eval_l2_loss']:.4f},\t"
                    f"eval aux {metrics_summary['eval_physics_loss']:.4f},\tseqs/s {metrics_summary['sequences_per_sec']:.1f},\t"
                    f"lr {train_summary['learning_rate']:.5f}"
                )

                last_eval_step = step

                if eval_summary["loss"] < best_eval_loss:
                    best_eval_loss = eval_summary["loss"]
                    if model_path:
                        torch.save(
                            {
                                "step": step,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": best_eval_loss,
                            },
                            model_path,
                        )
                        print(
                            f"New best model saved at step {step} with eval loss: {best_eval_loss:.4f}"
                        )

                tick = tock
                model.train()

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                metrics = forward(
                    model,
                    batch,
                    hparams["deltas_loss_weight"],
                    hparams["physics_loss_weight"],
                )
            metrics["loss"].backward()
            optimizer.step()
            train_metrics.append(
                {
                    k: v.detach().item() if v is not None else None
                    for k, v in metrics.items()
                }
            )
            scheduler.step()
            step += 1

            if step == total_steps:
                break

    print(f"Training completed after {total_steps} steps.")
    return all_metrics, model_path
