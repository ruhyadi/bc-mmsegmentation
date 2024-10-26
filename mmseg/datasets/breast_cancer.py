"""Breast cancer dataset."""

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class BreastCancerDataset(BaseSegDataset):
    """Breast cancer dataset."""

    METAINFO = dict(
        classes=(["background", "benign", "malignant"]),
        pallete=[[50, 50, 50], [100, 100, 100], [150, 150, 150]],
    )

    def __init__(
        self,
        img_suffix=".jpg",
        seg_map_suffix=".png",
        reduce_zero_label=False,
        **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )
