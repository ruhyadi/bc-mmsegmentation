"""Plot training graph."""

import rootutils

ROOT = rootutils.autosetup()

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm


def plot_train(
    scalars_path: str, log_path: str, output_path: str, downsample_n: int = 40
):
    """Plot training graph."""
    output_path: Path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    scalars: List[ScalarsSchema] = []
    metrics: List[MetricsSchema] = []
    with open(scalars_path, "r") as f:
        # read per line
        lines = f.readlines()
        for line in tqdm(lines, desc="Reading scalars"):
            line = json.loads(line)

            if "lr" in line:
                line = ScalarsSchema(
                    lr=line["lr"],
                    data_time=line["data_time"],
                    loss=line["loss"],
                    decode_loss_ce=line["decode.loss_ce"],
                    decode_loss_dice=line["decode.loss_dice"],
                    decode_acc_seg=line["decode.acc_seg"],
                    aux_loss_ce=line["aux.loss_ce"],
                    aux_acc_seg=line["aux.acc_seg"],
                    time=line["time"],
                    iter=line["iter"],
                    memory=line["memory"],
                    step=line["step"],
                )
                scalars.append(line)
            else:
                line = MetricsSchema(
                    aAcc=line["aAcc"],
                    mDice=line["mDice"],
                    mAcc=line["mAcc"],
                    data_time=line["data_time"],
                    time=line["time"],
                    step=line["step"],
                )
                metrics.append(line)

    # pandas dataframe
    print(f"Creating scalars and metrics dataframes")
    scalars_df = pd.DataFrame([s.model_dump() for s in scalars])
    metrics_df = pd.DataFrame([m.model_dump() for m in metrics])
    print(f"Scalars and metrics dataframes created")
    print(f"Total rows in scalars_df: {scalars_df.shape[0]}")

    # downsampling
    print(f"Downsampling scalars and metrics dataframes")
    scalars_df = scalars_df.iloc[::downsample_n, :]
    print(f"Total rows in scalars_df after downsampling: {scalars_df.shape[0]}")

    # save as csv
    scalars_df.to_csv(output_path / "scalars.csv", index=False)
    metrics_df.to_csv(output_path / "metrics.csv", index=False)


class ScalarsSchema(BaseModel):

    lr: float
    data_time: float
    loss: float
    decode_loss_ce: float
    decode_loss_dice: float
    decode_acc_seg: float
    aux_loss_ce: float
    aux_acc_seg: float
    time: float
    iter: int
    memory: int
    step: int


class MetricsSchema(BaseModel):

    aAcc: float
    mDice: float
    mAcc: float
    data_time: float
    time: float
    step: int


if __name__ == "__main__":

    plot_train(
        scalars_path="work_dirs/unet-s5-d16_deeplabv3_breast-cancer/20240928_152200/vis_data/scalars.json",
        log_path="work_dirs/unet-s5-d16_deeplabv3_breast-cancer/20240928_152200/20240928_152200.log",
        output_path="tmp/train_graph",
    )
