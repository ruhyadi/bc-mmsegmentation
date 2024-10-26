"""
Predict the image

usage:
python tools/dmid/predict.py \
    --config_path work_dirs/unet-s5-d16_deeplabv3_breast-cancer_3classes/unet-s5-d16_deeplabv3_breast-cancer.py \
    --weights_path work_dirs/unet-s5-d16_deeplabv3_breast-cancer_3classes/iter_40000.pth \
    --images_dir data/bc-dataset/tmp/data_final_cls/images/val \
    --output_dir tmp/starbuck \
    --metadata_csv_path data/bc-dataset/assets/metadata.csv
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
    metadata_csv_path: Optional[str] = None,
    resize_ratio: float = 0.25,
) -> None:
    """Predict the image."""
    images_dir: Path = Path(images_dir)
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir = output_dir / "vis"
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    biner_output_dir = output_dir / "biner"
    biner_output_dir.mkdir(parents=True, exist_ok=True)

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

        # get metadata
        if metadata_csv_path:
            meta = metadata.get_rows_by_filename(image_path.stem)

        # find contours from predictions
        contours, _ = cv2.findContours(
            image=preds,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        # draw contours
        mask = np.zeros_like(img)
        for contour in contours:
            target_cls = None
            if metadata_csv_path:
                for row in meta:
                    if (
                        row.center_xy is not None
                        and row.radius is not None
                        and cv2.pointPolygonTest(contour, tuple(row.center_xy), True)
                        > 0
                    ):
                        target_cls = row.class_abnormality.upper()
                        break

            # get contour parameter
            (x, y), (w, h), angle = cv2.fitEllipse(contour)
            radius = int(max(w, h) // 2)
            contour_color = preds[int(y), int(x)]

            color = (0, 255, 0)  # green
            pred_cls = classes[int(contour_color)].upper()
            pred_txt = f"Pred: {pred_cls}"

            if target_cls:
                color = (0, 255, 0) if pred_cls == target_cls else (0, 0, 255)
                pred_txt += f" | Target: {target_cls}"

            # draw contour
            if not metadata_csv_path or target_cls:
                cv2.drawContours(mask, [contour], -1, color, -1)
                img = cv2.addWeighted(img, 1, mask, 0.1, 0)
                cv2.drawContours(img, [contour], -1, color, 5)

                # draw text
                cv2.putText(
                    img,
                    pred_txt,
                    (int(x) - radius, int(y) - radius - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    color,
                    8,
                    cv2.LINE_AA,
                )

        # save image
        output_path = vis_output_dir / image_path.name
        img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
        cv2.imwrite(str(output_path), img)

        # save biner
        output_path = biner_output_dir / f"{image_path.stem}.png"
        preds = cv2.resize(preds, (0, 0), fx=resize_ratio, fy=resize_ratio) * 100
        cv2.imwrite(str(output_path), preds)

    print(f"Predicted images are saved at {output_dir}")


if __name__ == "__main__":
    """Main."""
    parser = argparse.ArgumentParser(description="Predict the image")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to the weights file",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Path to the images directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["BG", "B", "M"],
        help="Classes",
    )
    parser.add_argument(
        "--metadata_csv_path",
        type=str,
        default=None,
        help="Path to the metadata CSV",
    )
    parser.add_argument(
        "--resize_ratio",
        type=float,
        default=0.25,
        help="Resize ratio",
    )
    args = parser.parse_args()

    predict(
        config_path=args.config_path,
        weights_path=args.weights_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        device=args.device,
        classes=args.classes,
        metadata_csv_path=args.metadata_csv_path,
        resize_ratio=args.resize_ratio,
    )
