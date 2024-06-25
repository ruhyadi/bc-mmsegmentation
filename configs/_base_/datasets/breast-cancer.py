# dataset settings
dataset_type = "BreastCancerDataset"
data_root = "data/bc-dataset/tmp/final"
img_scale = (512, 512)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=img_scale, keep_ratio=False),
    dict(type="RandomFlip", prob=0.5),
    # dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=img_scale, keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
img_ratios = [1.0]
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type="RandomFlip", prob=0.0, direction="horizontal"),
                dict(type="RandomFlip", prob=1.0, direction="horizontal"),
            ],
            [dict(type="LoadAnnotations")],
            [dict(type="PackSegInputs")],
        ],
    ),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=10000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(img_path="images/train", seg_map_path="annotations/train"),
            pipeline=train_pipeline,
        ),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images/val", seg_map_path="annotations/val"),
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mDice"])
test_evaluator = val_evaluator
