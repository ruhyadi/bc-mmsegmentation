default_scope = "mmseg"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
# vis_backends = [dict(type="LocalVisBackend"), dict(type="WandbVisBackend")]
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = "work_dirs/unet-s5-d16_deeplabv3_breast-cancer_640x640_best/iter_40000.pth"
resume = False

tta_model = dict(type="SegTTAModel")
