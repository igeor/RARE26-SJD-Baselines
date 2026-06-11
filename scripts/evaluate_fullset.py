# Load an .npz file
import argparse
from pathlib import Path
import sys
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate_kfold import evaluate


def load_npz_file(file_path):
    data = np.load(file_path)
    y_true = data['y_true']
    y_scores = data['y_scores']
    return y_true, y_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", help="Directory containing prediction files for each fold")
    parser.add_argument("--epoch", type=int, default=15, help="Epoch number to evaluate (default: 15)")
    parser.add_argument("--eval_model", type=str, default="last", help="Evaluation model to use (e.g., 'last', 'best')")
    args = parser.parse_args()
    
    predictions_dir = Path(args.predictions_dir)
    if not predictions_dir.exists():
        print(f"Error: Directory {predictions_dir} does not exist.")
        sys.exit(1)
    
    if args.eval_model not in ["last", "best"]:
        print(f"Error: Invalid eval_model '{args.eval_model}'. Must be 'last' or 'best'. Using default: 'last'.")
        args.eval_model = "last"

    # Config in output dir 
    config_path = predictions_dir / "config.txt"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    file_path = rf"{args.predictions_dir}\epoch_{args.epoch}_full_predictions.npz"  # Replace with your .npz file path
    y_true, y_scores = load_npz_file(file_path)

    print("OOF Predictions:")
    y_true_oof = np.asarray(y_true).reshape(-1).astype(int)
    y_scores_oof = np.asarray(y_scores).reshape(-1).astype(float)
    print(evaluate(y_true=y_true_oof, y_scores=y_scores_oof, n_bootstrap=1000, seed=42))