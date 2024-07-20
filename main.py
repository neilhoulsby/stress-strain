import torch
import torch.optim as optim
import numpy as np
import time
from config import hparams
from data import get_datasets
from model import Transformer, TransformerConfig
from utils import forward, build_deltas, create_learning_rate_scheduler

def main():
    torch.cuda.empty_cache()
    torch.manual_seed(hparams["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, eval_loader = get_datasets(batch_size=hparams["batch_size"])

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
    model = Transformer(config).to(device)
    model.train()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        create_learning_rate_scheduler(
            warmup_steps=hparams["warmup_steps"], total_steps=hparams["total_steps"]
        ),
    )

    metrics_all = []
    total_steps = 0
    tick = time.time()

    while total_steps < hparams["total_steps"]:
        for batch in train_loader:
            if total_steps == 1 or (
                total_steps % hparams["eval_freq"] == 0 and total_steps > 0
            ):
                summary = {k: np.mean([m[k] for m in metrics_all]) for k in metrics_all[0]}
                summary["learning_rate"] = scheduler.get_last_lr()[0]
                metrics_all = []

                tock = time.time()
                steps_per_sec = hparams["eval_freq"] / (tock - tick)
                tick = tock

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

                print(
                    f"Step: {total_steps:04d},\ttrain loss {summary['loss']:.3f},\t"
                    f"train l2 {summary['l2_loss']:.3f},\ttrain aux {summary['physics_loss']:.3f},\t"
                    f"eval loss {eval_summary['loss']:.3f},\teval l2 {eval_summary['l2_loss']:.3f},\t"
                    f"eval aux {eval_summary['physics_loss']:.3f},\tsteps/s {steps_per_sec:.1f},\t"
                    f"lr {summary['learning_rate']:.5f}"
                )

                model.train()

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            metrics = forward(
                model,
                batch,
                hparams["deltas_loss_weight"],
                hparams["physics_loss_weight"],
            )
            metrics["loss"].backward()
            optimizer.step()
            metrics_all.append(
                {
                    k: v.detach().item() if v is not None else None
                    for k, v in metrics.items()
                }
            )
            scheduler.step()
            total_steps += 1

            if total_steps >= hparams["total_steps"]:
                break

    print(f"Training completed after {total_steps} steps.")

if __name__ == "__main__":
    main()
