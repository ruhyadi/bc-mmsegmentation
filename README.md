# MMSegmentation for Breast Cancer Research

## Version

The version of the MMSegmentation details are as follows:

| Version                                                                                                 | Date       | Description                                                                                               |
| ------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------- |
| [b040e14](https://github.com/open-mmlab/mmsegmentation/commit/b040e147adfa027bbc071b624bedf0ae84dfc922) | 2024-03-22 | The commit is based on version [v1.2.2](https://github.com/open-mmlab/mmsegmentation/releases/tag/v1.2.2) |

## Getting Started

### Prerequisites

We assume you have **docker** installed on you machine. If not, you can install it from [here](https://docs.docker.com/get-docker/). Also, you need to have installed the **NVIDIA Container Toolkit** to run the docker container with GPU support (at this moment we don't support it yet). You can install it from [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Development

For development, we will use [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) to create a development environment. So, you need to have **Visual Studio Code** installed on your machine. If not, you can install it from [here](https://code.visualstudio.com/). Also, you need to have the **Remote - Containers** extension installed on your Visual Studio Code. If not, you can install it from [here](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

Next, you need to clone the repository to your local machine by:
```bash
git clone https://github.com/ruhyadi/bc-mmsegmentation
```

You can choose the devcontainer by pressing `F1` and type `Remote-Containers: Rebuild Container` (or `Reopen in Container`). Then, choose the devcontainer you want to use.

After that, you will directly enter the devcontainer and you can start developing the project.

### Dataset Preparation

Dataset for this project can be found at **Digital mammography Dataset for Breast Cancer Diagnosis Research** [Figshare DMID](https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883). All you need to do is only download the following files:

- `TIFF Images.zip`: Contains the original images in TIFF format. (4.32GB).

You also need to download the ground truth data from [Github Release](https://github.com/ruhyadi/bc-mmsegmentation/releases). Download only the following files:

-  `DMID_Ground_Truth.zip`: Contains the ground truth data for the images in PNG format. (2.0MB). [Download](https://github.com/ruhyadi/bc-mmsegmentation/releases/download/v1.0/DMID_Ground_Truth.zip).
-  `DMID_metadata.zip`: Contains the metadata for the images in CSV format. (1.0MB). [Download](https://github.com/ruhyadi/bc-mmsegmentation/releases/download/v1.0/DMID_metadata.zip)

Next, you need to extract the files and put them in the following directory structure:

```bash
data
├── DMID
│   ├── ground_truth
│   │   ├── IMG001.png
│   │   ├── IMG002.png
│   │   ├── ...
│   │   └── IMG502.png
│   ├── metadata
│   │   ├── metadata.csv
│   └── tiff_images
│       ├── IMG001.tif
│       ├── IMG002.tif
│       ├── ...
│       └── IMG510.tif
```

Next, you need to generate training and validation data by running the following command:

```bash
python tools/prepare_dataset.py \
    --images_dir data/dmid/tiff_images \
    --anns_dir data/dmid/ground_truth \
    --metadata_csv_path data/dmid/metadata/metadata.csv \
    --output_dir data/dmid/training \
    --categories B M \
    --resize_ratio 1.00 \
    --train_ratio 0.8
```

The script will generate the following directory structure:

```bash
data
├── DMID
│   ├── ground_truth
│   ├── metadata
│   ├── tiff_images
│   ├── training
│   │   ├── images
│   │   │   ├── train
│   │   │   │   ├── IMG001.jpg
│   │   │   │   ├── IMG002.jpg
│   │   │   │   ├── ...
│   │   │   │   └── IMG401.jpg
│   │   │   └── val
│   │   │       ├── IMG402.jpg
│   │   │       ├── IMG403.jpg
│   │   │       ├── ...
│   │   │       └── IMG502.jpg
│   │   ├── annotations
│   │   │   ├── train
│   │   │   │   ├── IMG001.png
│   │   │   │   ├── IMG002.png
│   │   │   │   ├── ...
│   │   │   │   └── IMG401.png
│   │   │   └── val
│   │   │       ├── IMG402.png
│   │   │       ├── IMG403.png
│   │   │       ├── ...
│   │   │       └── IMG502.png
```

We will use the `data/DMID/training` directory for training the model.

### Training

Training segmentation models with MMSegmentation is straightforward. You only need to do the following steps:

##### 1. Create dataset module

Create a dataset module in `mmseg/datasets` directory. In this project, we have created the `breast_cancer.py` dataset module. You can see the details in the [mmseg/datasets/breast_cancer.py](mmseg/datasets/breast_cancer.py).

##### 2. Create dataset configuration

Create a dataset configuration in `configs/_base_/datasets` directory. In this project, we have created the `breast_cancer.py` dataset configuration. You can see the details in the [configs/_base_/datasets/breast_cancer.py](configs/_base_/datasets/breast-cancer.py).

##### 3. Create model configuration

Create a model configuration in `configs/_base_/models` directory. In this project, we have created the `unet-s5-d16_deeplabv3_breast-cancer.py` model configuration. You can see the details in the [configs/_base_/models/unet-s5-d16_deeplabv3_breast-cancer.py](configs/unet/unet-s5-d16_deeplabv3_breast-cancer.py).

### UNET with DeepLabV3

## Acknowledgement

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab Semantic Segmentation Toolbox and Benchmark.