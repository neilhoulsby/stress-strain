# python -m main --note 'My experiment' [--debug]
import argparse
import csv
import datetime
import os
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
    print("Running hyperparameter sweep...")

    if not args.debug:
        # Create timestamped directory for this experiment
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join("results", timestamp)
        os.makedirs(experiment_dir, exist_ok=True)

        # Save the note
        note_path = os.path.join(experiment_dir, "note.txt")
        with open(note_path, "w") as f:
            f.write(args.note)
        print(f"Experiment note saved to {note_path}")
    else:
        print("Debug mode: No directories will be created and no data will be saved.")

    all_results = []
    for run_index, hparams in enumerate(generate_sweep_configs(), start=1):
        if not args.debug:
            # Create subdirectory for this run
            run_dir = os.path.join(experiment_dir, str(run_index))
            os.makedirs(run_dir, exist_ok=True)

            # Update model_dir in hparams
            hparams["model_dir"] = run_dir
        else:
            hparams["model_dir"] = None  # or a temporary directory if needed

        print(f"Run {run_index}: Training with hyperparameters: {hparams}")
        metrics, best_model_path = train(hparams)

        # Process results
        run_results = []
        for metric in metrics:
            flat_hparams = flatten_dict(hparams)
            result = {
                "run_index": run_index,
                **flat_hparams,
                **metric,
            }  # Add run_index to each result
            run_results.append(result)

        all_results.extend(run_results)

        if not args.debug:
            # Save run-specific results
            run_csv_path = os.path.join(run_dir, f"results_{run_index}.csv")
            save_results_to_csv(run_results, run_csv_path)
            print(f"Run {run_index} results saved to {run_csv_path}")
            print(f"Run {run_index} best model saved at: {best_model_path}")
        else:
            print(f"Debug mode: Run {run_index} results and model not saved.")

    if not args.debug:
        # Save all results to a single CSV in the experiment directory
        all_results_path = os.path.join(experiment_dir, "all_results.csv")
        save_results_to_csv(all_results, all_results_path)
        print(f"All results saved to {all_results_path}")
    else:
        print("Debug mode: All results not saved.")


def save_results_to_csv(results, filename):
    if results:
        keys = results[0].keys()
        with open(filename, "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep for training"
    )
    parser.add_argument(
        "--note",
        type=str,
        required=True,
        help="A note about the experiment, saved as note.txt in the experiment directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: no directories created, no data saved",
    )
    args = parser.parse_args()
    main(args)
