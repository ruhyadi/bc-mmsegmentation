"""Breast cancer dataset."""

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class BreastCancerDataset(BaseSegDataset):
    """Breast cancer dataset."""

    METAINFO = dict(
        classes=(["tumor"]),
        pallete=[[100, 100, 100]],
    )

    def __init__(
        self, img_suffix=".jpg", seg_map_suffix=".png", reduce_zero_label=True, **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )
