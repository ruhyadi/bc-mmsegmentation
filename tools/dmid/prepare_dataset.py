"""
Prepare dataset for mmsegmentation
usage:
python tools/prepare_dataset.py \
    --images_dir data/dmid/tiff_images \
    --anns_dir data/dmid/ground_truth \
    --metadata_csv_path data/dmid/metadata/metadata.csv \
    --output_dir data/dmid/final \
    --categories B M \
    --resize_ratio 1.00 \
    --train_ratio 0.8
"""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import argparse
from pathlib import Path
from tqdm import tqdm
import random

import cv2
import numpy as np

from tools.dmid.metadata_schema import Metadata


class PrepareDataset:
    """Prepare dataset for mmsegmentation."""

    def __init__(
        self,
        images_dir: str,
        anns_dir: str,
        metadata_csv_path: str,
        output_dir: str,
        categories: List[str] = ["B", "M"],
        resize_ratio: float = 1.00,
        train_ratio: float = 0.8,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.anns_dir = Path(anns_dir)
        self.metadata_csv_path = Path(metadata_csv_path)
        self.output_dir = Path(output_dir)
        self.categories = categories
        self.resize_ratio = resize_ratio
        self.train_ratio = train_ratio

        # create new directories
        self._create_output_dirs()

        # metadata
        self.metadata = Metadata()
        self.metadata.load_csv(self.metadata_csv_path)

    def setup(self) -> None:
        """Prepare dataset."""
        print(f"Converting dataset to MMsegmentation format...")

        # split by annotations
        anns = list(self.anns_dir.glob("*"))
        random.shuffle(anns)
        num_train = int(len(anns) * self.train_ratio)
        train_anns = anns[:num_train]
        val_anns = anns[num_train:]

        # convert training data
        for ann_path in tqdm(train_anns, desc="Converting training data..."):
            img_path = self.images_dir / ann_path.name.replace(".png", ".tif")
            if not img_path.exists():
                raise FileNotFoundError(f"File not found: {img_path}")
            self.tif_to_jpg(
                tif_path=str(img_path),
                output_path=str(self.new_images_train_dir / img_path.name),
                resize_ratio=self.resize_ratio,
            )
            self.setup_ann(
                ann_path=str(ann_path),
                output_path=str(self.new_anns_train_dir / ann_path.name),
                categories=self.categories,
            )

        # convert validation data
        for ann_path in tqdm(val_anns, desc="Converting validation data..."):
            img_path = self.images_dir / ann_path.name.replace(".png", ".tif")
            if not img_path.exists():
                raise FileNotFoundError(f"File not found: {img_path}")
            self.tif_to_jpg(
                tif_path=str(img_path),
                output_path=str(self.new_images_val_dir / img_path.name),
                resize_ratio=self.resize_ratio,
            )
            self.setup_ann(
                ann_path=str(ann_path),
                output_path=str(self.new_anns_train_dir / ann_path.name),
                categories=self.categories,
            )

        print(f"Dataset is ready at {self.output_dir}")
        print(f"Training size: {len(train_anns)}")
        print(f"Validation size: {len(val_anns)}")

    def tif_to_jpg(
        self, tif_path: str, output_path: str, resize_ratio: float
    ) -> np.ndarray:
        """Convert tif to jpg."""
        img = cv2.imread(tif_path, cv2.IMREAD_COLOR)  # BGR
        img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)
        cv2.imwrite(output_path.replace(".tif", ".jpg"), img)

    def setup_ann(self, ann_path: str, output_path: str, categories: List[str]) -> None:
        """Setup annotation."""
        mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)  # grayscale
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # convert mask to categories index
        filename = ann_path.split("/")[-1].split(".")[0]
        metas = self.metadata.get_rows_by_filename(filename)
        for meta in metas:
            for contour in contours:
                if (
                    meta.center_xy is not None
                    and meta.radius is not None
                    and cv2.pointPolygonTest(contour, tuple(meta.center_xy), True) > 0
                ):
                    cv2.drawContours(
                        mask,
                        [contour],
                        -1,
                        (categories.index(meta.class_abnormality) + 1),
                        -1,
                    )

        cv2.imwrite(output_path, mask)

    def _create_output_dirs(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.new_images_train_dir = self.output_dir / "images" / "train"
        self.new_images_val_dir = self.output_dir / "images" / "val"
        self.new_anns_train_dir = self.output_dir / "annotations" / "train"
        self.new_anns_val_dir = self.output_dir / "annotations" / "val"

        self.new_images_train_dir.mkdir(parents=True, exist_ok=True)
        self.new_images_val_dir.mkdir(parents=True, exist_ok=True)
        self.new_anns_train_dir.mkdir(parents=True, exist_ok=True)
        self.new_anns_val_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for mmsegmentation")
    parser.add_argument(
        "--images_dir", type=str, required=True, help="Input images directory"
    )
    parser.add_argument(
        "--anns_dir", type=str, required=True, help="Input annotations directory"
    )
    parser.add_argument(
        "--metadata_csv_path", type=str, required=True, help="Metadata CSV path"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["B", "M"],
        help="Categories to be converted",
    )
    parser.add_argument("--resize_ratio", type=float, default=1.00, help="Resize ratio")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training ratio")
    args = parser.parse_args()

    prepare_dataset = PrepareDataset(
        images_dir=args.images_dir,
        anns_dir=args.anns_dir,
        metadata_csv_path=args.metadata_csv_path,
        output_dir=args.output_dir,
        categories=args.categories,
        resize_ratio=args.resize_ratio,
        train_ratio=args.train_ratio,
    )
    prepare_dataset.setup()
