_base_ = [
    "../_base_/models/deeplabv3_unet_s5-d16.py",
    "../_base_/datasets/breast-cancer.py",
    "../_base_/default_runtime.py",
    # "../_base_/schedules/schedule_40k.py",
    "../_base_/schedules/schedule_20k.py",
]
crop_size = (640, 640)
stride = (341, 341)
data_preprocessor = dict(size=crop_size)
model = dict(
data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=crop_size, stride=stride),
    decode_head=dict(
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=1.0),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=3.0),
        ],
        num_classes=3,
    ),
    auxiliary_head=dict(num_classes=3),
)
