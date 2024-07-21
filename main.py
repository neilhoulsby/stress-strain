import argparse
import csv
import datetime
from config import base_hparams
from config import generate_sweep_configs
from train import train


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main(args):
    if args.single_run_save_ckpt:
        print("Running a single training run with checkpoint saving...")
        hparams = base_hparams.copy()
        hparams['save_checkpoint'] = True
        metrics, best_model_path = train(hparams)

        # Save the results
        results = []
        for metric in metrics:
            flat_hparams = flatten_dict(hparams)
            result = {**flat_hparams, **metric}
            results.append(result)

        print(f"Best model saved at: {best_model_path}")
    else:
        print("Running hyperparameter sweep...")
        results = []
        for hparams in generate_sweep_configs():
            hparams['save_checkpoint'] = False
            print(f"Training with hyperparameters: {hparams}")
            metrics, _ = train(hparams)

            for metric in metrics:
                flat_hparams = flatten_dict(hparams)
                result = {**flat_hparams, **metric}
                results.append(result)

    # Save results to CSV (keep your existing code here)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.csv"

    if results:
        keys = results[0].keys()
        with open(filename, "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

        print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training with or without hyperparameter sweep"
    )
    parser.add_argument(
        "--single_run_save_ckpt",
        action="store_true",
        help="If set, runs a single training session and saves the best checkpoint",
    )
    args = parser.parse_args()

    main(args)
