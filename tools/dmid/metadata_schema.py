"""Metadata schema."""

import rootutils

ROOT = rootutils.autosetup()

import csv
from typing import List, Optional

from pydantic import BaseModel, Field


class Metadata(BaseModel):

    rows: List["MetadataRow"] = Field([])

    def load_csv(self, csv_path: str) -> None:
        """Load metadata from CSV."""
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(
                    MetadataRow(
                        filename=row["filename"],
                        mammogram_view=row["mammogram_view"],
                        background=row["background"],
                        abnormality=row["abnormality"],
                        class_abnormality=row.get("class_abnormality") or None,
                        center_xy=(
                            [int(row["x"]), int(row["y"])]
                            if row.get("x") and row.get("y")
                            else None
                        ),
                        radius=int(row["radius"]) if row.get("radius") else None,
                    )
                )

    def get_rows_by_filename(self, filename: str) -> List["MetadataRow"]:
        """Get rows by filename."""

        return [row for row in self.rows if row.filename == filename]

    def _cxcy_to_xywh(self, cxcy: List[int], radius: int) -> List[int]:
        """Convert center and radius to xywh."""
        cx, cy = cxcy
        return [cx - radius, cy - radius, radius * 2, radius * 2]

    def _radius_to_area(self, radius: int) -> float:
        """Convert radius to area."""
        return 3.14 * radius**2


class MetadataRow(BaseModel):

    filename: str = Field(..., example="IMG001")
    mammogram_view: str = Field(..., example="MLOLT")
    background: str = Field(..., example="G")
    abnormality: str = Field(..., example="MISC+CALC")
    class_abnormality: Optional[str] = Field(None, example="M")
    center_xy: Optional[List[int]] = Field(None, example=[100, 200])
    radius: Optional[int] = Field(None, example=50)
