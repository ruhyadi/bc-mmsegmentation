import rootutils

ROOT = rootutils.autosetup()

from pathlib import Path
from tqdm import tqdm

from mmseg.apis import init_model, inference_model, show_result_pyplot, MMSegInferencer


def demo():
    """Demo script."""
    config_path = "work_dirs/unet-s5-d16_deeplabv3_breast-cancer_1500x1188_512x512/unet-s5-d16_deeplabv3_breast-cancer.py"
    checkpoint_path = (
        "work_dirs/unet-s5-d16_deeplabv3_breast-cancer_1500x1188_512x512/iter_10000.pth"
    )

    model = init_model(config_path, checkpoint_path, "cpu")

    # imgs_path = "data/bc-dataset/tmp/data_final/images/val"
    imgs_path = "data/bc-dataset/tmp/data_inference"
    imgs_path = Path(imgs_path)

    output_dir = "tmp/breast-cancer-2"
    output_dir = Path(output_dir)

    for img_path in tqdm(sorted(list(imgs_path.glob("*.jpg")))):
        result = inference_model(model, img_path)
        print(f"Result: {result.gt_sem_seg}")
        vis_image = show_result_pyplot(
            model,
            str(img_path),
            result,
            out_file=str(output_dir / img_path.name),
            show=False,
            save_dir=str(output_dir),
            draw_gt=False,
        )


def demo2():
    import cv2

    config_path = "work_dirs/unet-s5-d16_deeplabv3_breast-cancer/unet-s5-d16_deeplabv3_breast-cancer.py"
    checkpoint_path = (
        "work_dirs/unet-s5-d16_deeplabv3_breast-cancer/iter_40000.pth"
    )

    inferencer = MMSegInferencer(
        config_path,
        checkpoint_path,
        device="cuda:0",
        classes=["background", "tumor"],
        palette=[[0, 0, 0], [255, 255, 255]],
    )

    # imgs_path = "data/bc-dataset/tmp/final/images/val"
    imgs_path = "data/bc-dataset/tmp/data_final/images/val"
    imgs_path = Path(imgs_path)

    # output_dir = "tmp/breast-cancer-14-dice-57-40k"
    output_dir = "tmp/breast-cancer-640x640-2-val"
    output_dir = Path(output_dir)

    for i, img_path in tqdm(enumerate(sorted(list(imgs_path.glob("*.jpg"))))):
        inferencer(str(img_path), show=False, out_dir=str(output_dir))
        result_img = output_dir / "pred" / f"{i:08d}_pred.png"
        img = cv2.imread(str(result_img), cv2.IMREAD_GRAYSCALE)
        img[img == 1] = 255
        output_file = output_dir / "pred" / Path(img_path).name
        cv2.imwrite(str(output_file), img)

if __name__ == "__main__":
    # demo()
    demo2()
