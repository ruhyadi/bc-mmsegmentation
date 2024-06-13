
import rootutils

ROOT = rootutils.autosetup()

from pathlib import Path
from tqdm import tqdm

from mmseg.apis import init_model, inference_model, show_result_pyplot


def demo():
    """Demo script."""
    config_path = "work_dirs/unet-s5-d16_deeplabv3_breast-cancer_1500x1188_512x512/unet-s5-d16_deeplabv3_breast-cancer.py"
    checkpoint_path = "work_dirs/unet-s5-d16_deeplabv3_breast-cancer_1500x1188_512x512/iter_10000.pth"

    model = init_model(config_path, checkpoint_path, "cuda:0")

    imgs_path = "data/bc-dataset/tmp/data_final/images/val"
    imgs_path = Path(imgs_path)

    output_dir = "tmp/breast-cancer"
    output_dir = Path(output_dir)

    for img_path in tqdm(sorted(list(imgs_path.glob("*.jpg")))):
        result = inference_model(model, img_path)
        vis_image = show_result_pyplot(
            model,
            str(img_path),
            result,
            out_file=str(output_dir / img_path.name),
            show=False,
            save_dir=str(output_dir),
        )

if __name__ == "__main__":
    demo()