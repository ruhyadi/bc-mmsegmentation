"""
Extract metrics from log file

usage:
python tools/metrics_from_log.py \
    --log_file /path/to/log_file.log \
    --categories background benign malignant \
    --output_dir /path/to/output_dir
"""

import rootutils

ROOT = rootutils.autosetup()

import argparse

from typing import List
from pathlib import Path
import shutil


def metrics_from_log(
    log_file: str,
    output_dir: str,
    categories: List[str] = ["background", "benign", "malignant"],
) -> None:
    """Extract metrics from log file."""
    log_file: Path = Path(log_file)
    output_dir: Path = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    with open(str(output_dir / "metrics.csv"), "w") as f:
        cats_metrics = [
            f"{cat}_{metric}" for cat in categories for metric in ["dice", "acc"]
        ]
        f.write("step," + ",".join(cats_metrics) + ",mDice,mAcc\n")


    with open(log_file, "r") as f:
        lines = f.readlines()

        last_step = None
        cat_metrics = {cat: {"dice": None, "acc": None} for cat in categories}
        mDice = None
        mAcc = None

        for line in lines:
            
            if "Iter(train)" in line:
                # INFO - Iter(train) [  500/40000]  lr: 9.8888e-03
                last_step = int(line.split("[")[1].split("]")[0].split("/")[0])

            if "|" in line:
                # | background | 98.77 | 97.98 |
                for cat in categories:
                    if cat in line:
                        cat_metrics[cat]["dice"] = float(line.split("|")[2])
                        cat_metrics[cat]["acc"] = float(line.split("|")[3])
                        break


            if "mDice:" in line:
                # Iter(val) [215/215]    aAcc: 97.2500  mDice: 44.8800
                mDice = float(line.split("mDice:")[1].split("mAcc:")[0])
                mAcc = float(line.split("mAcc:")[1].split("data_time:")[0])

            if last_step and mDice:
                with open(str(output_dir / "metrics.csv"), "a") as f:
                    metrics_values = [str(last_step)]
                    for cat in categories:
                        metrics_values.append(str(cat_metrics[cat]["dice"]))
                        metrics_values.append(str(cat_metrics[cat]["acc"]))
                    metrics_values.append(str(mDice))
                    metrics_values.append(str(mAcc))
                    f.write(",".join(metrics_values) + "\n")

                # reset
                last_step = None
                mDice = None
                mAcc = None
                cat_metrics = {cat: {"dice": None, "acc": None} for cat in categories}

    print(f"Metrics saved to {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract metrics from log file")
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="Path to log file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["background", "benign", "malignant"],
        help="Categories",
    )
    args = parser.parse_args()
    metrics_from_log(args.log_file, args.output_dir, args.categories)
