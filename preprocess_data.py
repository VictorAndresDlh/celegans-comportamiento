from pathlib import Path

from utils_data import write_preprocessed_csv


if __name__ == "__main__":
    root = Path("Datos")
    out = Path("Analysis/Preprocessed")

    print(f"Preprocessing data from {root} -> {out} ...")
    all_traj_path, worm_summary_path = write_preprocessed_csv(root, out)
    print("Saved:")
    print(f"  - {all_traj_path}")
    print(f"  - {worm_summary_path}")
