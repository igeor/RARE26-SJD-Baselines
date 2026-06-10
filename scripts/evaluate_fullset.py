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
    args = parser.parse_args()
    
    predictions_dir = Path(args.predictions_dir)
    if not predictions_dir.exists():
        print(f"Error: Directory {predictions_dir} does not exist.")
        sys.exit(1)

    # Config in output dir 
    config_path = predictions_dir / "config.txt"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    n_folds = config.get("k", 5)  # Default to 5 folds if not specified
    
    y_true_oof, y_scores_oof = [], []

    for fold_idx in range(1, n_folds + 1):
        
        file_path = rf"{args.predictions_dir}\fold_{fold_idx}\best_val_predictions.npz"  # Replace with your .npz file path
        y_true, y_scores = load_npz_file(file_path)
        y_true_oof.extend(y_true)
        y_scores_oof.extend(y_scores)

    print("OOF Predictions:")
    y_true_oof = np.asarray(y_true_oof).reshape(-1).astype(int)
    y_scores_oof = np.asarray(y_scores_oof).reshape(-1).astype(float)
    print(evaluate(y_true=y_true_oof, y_scores=y_scores_oof, n_bootstrap=1000, seed=42))