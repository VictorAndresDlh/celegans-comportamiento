"""Shared data loading and preprocessing utilities for C. elegans trajectories.

This module centralizes filename parsing, data loading and preprocessing
so that all methodologies share a single, consistent implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import re


STRAINS = ["N2", "NL5901", "UA44", "TJ356", "BR5270", "BR5271", "ROSELLA"]


def parse_filename(file_path: Path | str) -> Tuple[Optional[str], Optional[str]]:
    """Parse strain and treatment from file path.

    Rules (agreed with user):
    - Strain: taken from directory name if it contains one of STRAINS; otherwise
      from pattern "<number>_<STRAIN>_..." in the filename.
    - Treatments:
      * CONTROL -> "Control"
      * ETANOL (anywhere in filename) and not ET 1:1000 -> "ETANOL" (pure ethanol)
      * ET 1:1000 variants ("ET1.1000", "ET_1_1000", etc.) are TOTAL EXTRACT and
        must be labeled as "Total_Extract_<compound>", where <compound> is
        inferred from folder/filenames: "CBD" or "CBDV".
      * CBD / CBDV doses map to:
          "CBD 0.3uM", "CBD 3uM", "CBD 30uM",
          "CBDV 0.3uM", "CBDV 3uM", "CBDV 30uM".
      * Anything else that cannot be parsed -> "Unknown".
    """
    p = Path(file_path)
    file_name = p.stem
    dir_name = p.parent.name

    # Strain from directory name, if possible
    strain = None
    for s in STRAINS:
        if s in dir_name:
            strain = s
            break

    # Fallback: strain from filename like "1234_BR5270_CONTROL.csv"
    if not strain:
        m = re.match(r"^\d+_(?P<strain>[A-Za-z0-9]+)_", file_name)
        if m:
            strain = m.group("strain")

    file_upper = file_name.upper()
    dir_upper = dir_name.upper()

    treatment: Optional[str] = None

    # 1) CONTROL
    if "CONTROL" in file_upper:
        treatment = "Control"

    # 2) Pure ETANOL (not total extract)
    elif "ETANOL" in file_upper and "ET" not in file_upper:
        # Clearly labeled ethanol without ET 1:1000 pattern
        treatment = "ETANOL"

    # 3) TOTAL EXTRACT (ET 1:1000 variants)
    elif "ET" in file_upper and (
        "1000" in file_upper or "1:1000" in file_upper or "1.1000" in file_upper or "1.100" in file_upper
    ):
        # Infer compound from folder or other csvs in folder
        folder_compound: Optional[str] = None
        if "_CBDV" in dir_upper:
            folder_compound = "CBDV"
        elif "_CBD" in dir_upper and "_CBDV" not in dir_upper:
            folder_compound = "CBD"

        if folder_compound is None:
            try:
                for csv_file in p.parent.glob("*.csv"):
                    name_u = csv_file.stem.upper()
                    if "CBDV" in name_u:
                        folder_compound = "CBDV"
                        break
                    if "CBD" in name_u and "CBDV" not in name_u:
                        folder_compound = "CBD"
                        break
            except Exception:
                pass

        if folder_compound is not None:
            treatment = f"Total_Extract_{folder_compound}"
        else:
            # We know it's a total extract but cannot attribute CBD vs CBDV
            treatment = "Total_Extract_Unknown"

    # 4) CBDV doses
    elif "CBDV" in file_upper:
        if "30UM" in file_upper or "30 UM" in file_upper:
            treatment = "CBDV 30uM"
        elif "3UM" in file_upper and not any(x in file_upper for x in ["30", "0.3", "0,3"]):
            treatment = "CBDV 3uM"
        elif "0,3" in file_upper or "0.3" in file_upper:
            treatment = "CBDV 0.3uM"
        else:
            treatment = "CBDV"

    # 5) CBD doses (no CBDV)
    elif "CBD" in file_upper and "CBDV" not in file_upper:
        if "30UM" in file_upper or "30 UM" in file_upper:
            treatment = "CBD 30uM"
        elif "3UM" in file_upper and not any(x in file_upper for x in ["30", "0.3", "0,3"]):
            treatment = "CBD 3uM"
        elif "0,3" in file_upper or "0.3" in file_upper:
            treatment = "CBD 0.3uM"
        else:
            treatment = "CBD"

    else:
        # Fallback: whatever comes after strain in filename
        parts = file_name.split("_")
        if len(parts) > 2:
            treatment = " ".join(parts[2:]).strip()
        else:
            treatment = "Unknown"

    return strain, treatment


def load_and_combine_tracking_data(file_paths: Sequence[Path | str]) -> Optional[pd.DataFrame]:
    """Load and combine tracking data, assigning per-file track IDs.

    Returns a DataFrame with at least columns:
      - experiment_id, file_name, strain, treatment,
      - track_id, worm_local_id, frame, x, y
    """
    records: List[pd.DataFrame] = []

    for i, path in enumerate(file_paths):
        p = Path(path)
        try:
            df = pd.read_csv(p, sep=";", usecols=["ID", "FRAME", "X", "Y"])
        except Exception as e:
            print(f"Could not read {p}: {e}")
            continue

        strain, treatment = parse_filename(p)
        experiment_id = p.stem.split("_")[0]

        df = df.rename(columns={"ID": "worm_local_id", "FRAME": "frame", "X": "x", "Y": "y"})
        df["file_name"] = p.name
        df["experiment_id"] = experiment_id
        df["strain"] = strain
        df["treatment"] = treatment
        # Technical unique identifier for each trajectory (worm within a file)
        df["track_id"] = f"{experiment_id}_{i}_" + df["worm_local_id"].astype(str)

        df = df.dropna(subset=["x", "y"])
        records.append(df)

    if not records:
        return None

    combined = pd.concat(records, ignore_index=True)
    return combined


def load_data_for_strain(root_data_dir: Path | str, target_strain: str,
                         exclude_unknown: bool = True) -> Dict[str, pd.DataFrame]:
    """Load all CSVs for a given strain, grouped by treatment.

    Returns {treatment: DataFrame} using the standardized columns from
    `load_and_combine_tracking_data`.
    """
    root = Path(root_data_dir)
    all_csv = list(root.rglob("*.csv"))

    by_treatment: Dict[str, List[Path]] = {}
    for fp in all_csv:
        strain, treatment = parse_filename(fp)
        if strain != target_strain:
            continue
        if exclude_unknown and (treatment is None or treatment == "Unknown"):
            continue
        if treatment not in by_treatment:
            by_treatment[treatment] = []
        by_treatment[treatment].append(fp)

    result: Dict[str, pd.DataFrame] = {}
    for treatment, paths in by_treatment.items():
        df = load_and_combine_tracking_data(paths)
        if df is not None and not df.empty:
            result[treatment] = df

    return result


def write_preprocessed_csv(root_data_dir: Path | str,
                           output_dir: Path | str) -> Tuple[Path, Path]:
    """Scan all data and write standardized preprocessed CSVs.

    - all_trajectories.csv : one row per frame
    - worm_summary.csv     : one row per worm
    """
    root = Path(root_data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_csv = list(root.rglob("*.csv"))
    combined = load_and_combine_tracking_data(all_csv)
    if combined is None or combined.empty:
        raise RuntimeError("No data found to preprocess")

    # All trajectories table
    all_traj_path = out / "all_trajectories.csv"
    combined.to_csv(all_traj_path, index=False)

    # Worm-level (track-level) summary
    grouped = combined.groupby("track_id")
    summary = grouped.agg(
        worm_local_id=("worm_local_id", "first"),
        strain=("strain", "first"),
        treatment=("treatment", "first"),
        experiment_id=("experiment_id", "first"),
        file_name=("file_name", "first"),
        n_frames=("frame", "count"),
        min_frame=("frame", "min"),
        max_frame=("frame", "max"),
        has_missing_positions=("x", lambda x: bool(x.isna().any())),
    ).reset_index()
    summary["duration_frames"] = summary["max_frame"] - summary["min_frame"]

    worm_summary_path = out / "worm_summary.csv"
    summary.to_csv(worm_summary_path, index=False)

    return all_traj_path, worm_summary_path
