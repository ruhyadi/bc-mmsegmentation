"""
Predict the image

usage:
python tools/dmid/predict.py \
"""

from typing import List, Optional
import rootutils

ROOT = rootutils.autosetup()

import argparse
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

from mmseg.apis.mmseg_inferencer import MMSegInferencer
from tools.dmid.metadata_schema import Metadata


def predict(
    config_path: str,
    weights_path: str,
    images_dir: str,
    output_dir: str,
    device: str = "cuda:0",
    classes: List[str] = ["BG", "B", "M"],
    gt_dir: Optional[str] = None,
    metadata_csv_path: Optional[str] = None,
    resize_ratio: float = 0.25,
) -> None:
    """Predict the image."""
    images_dir: Path = Path(images_dir)
    output_dir: Path = Path(output_dir)
    gt_dir: Path = Path(gt_dir) if gt_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    # metadata
    metadata = Metadata()
    if metadata_csv_path:
        metadata.load_csv(metadata_csv_path)

    # load model
    inferencer = MMSegInferencer(
        config_path,
        weights_path,
        device=device,
        classes=classes,
        palette=[[0, 0, 0], [100, 100, 100], [200, 200, 200]],
    )

    image_paths = sorted(list(images_dir.glob("*.jpg")))

    for image_path in tqdm(image_paths, desc="Predict"):
        img = cv2.imread(str(image_path))
        result = inferencer(str(image_path), show=False)
        preds = result["predictions"].astype(np.uint8)

        # find contours from predictions
        contours, _ = cv2.findContours(
            image=preds,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        # draw contours
        mask = np.zeros_like(img)
        for contour in contours:
            # get contour parameter
            (x, y), (w, h), angle = cv2.fitEllipse(contour)
            radius = int(max(w, h) // 2)
            contour_color = preds[int(y), int(x)]
            pred_txt = f"Pred: {classes[int(contour_color)].upper()}"

            # draw contour
            cv2.drawContours(mask, [contour], -1, (0, 255, 0), -1)
            img = cv2.addWeighted(img, 1, mask, 0.1, 0)
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 5)

            # draw text
            cv2.putText(
                img,
                pred_txt,
                (int(x) - radius, int(y) - radius - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 0),
                8,
                cv2.LINE_AA,
            )

        # save image
        output_path = output_dir / image_path.name
        img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
        cv2.imwrite(str(output_path), img)

        break


if __name__ == "__main__":
    predict(
        config_path="work_dirs/unet-s5-d16_deeplabv3_breast-cancer_3classes/unet-s5-d16_deeplabv3_breast-cancer.py",
        weights_path="work_dirs/unet-s5-d16_deeplabv3_breast-cancer_3classes/iter_40000.pth",
        images_dir="data/bc-dataset/tmp/data_final_cls/images/val",
        output_dir="tmp/starbuck",
    )
