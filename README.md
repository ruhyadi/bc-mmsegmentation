# MMSegmentation for Breast Cancer Research

## Introduction

This project is a research project to develop a semantic segmentation model for breast cancer diagnosis. The project uses the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) library to train and evaluate the segmentation model. The dataset used in this project is the **Digital mammography Dataset for Breast Cancer Diagnosis Research (DMID)** [Figshare DMID](https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883).

## Models

The models listed below are the segmentation models that have been trained and evaluated in DMID dataset. The models can be found in the [Github Release](https://github.com/ruhyadi/bc-mmsegmentation/releases). The models are trained using the following configurations:

| Model Name            | Backbone | Head      | Classes                                 | mDice | mAcc  | aAcc  | Config                                                        | Weight                                                                                                                             |
| --------------------- | -------- | --------- | --------------------------------------- | ----- | ----- | ----- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| UNET-S5-D16-DeepLabV3 | UNET     | DeepLabV3 | 3 (`background`, `benign`, `malignant`) | 98.58 | 98.4  | 99.94 | [config](configs/unet/unet-s5-d16_deeplabv3_breast-cancer.py) | [Download](https://github.com/ruhyadi/bc-mmsegmentation/releases/download/v1.0/unet-s5-d16_deeplabv3_breast-cancer_2024-09-28.zip) |
| UNET-S5-D16-DeepLabV3 | UNET     | DeepLabV3 | 2 (`background`, `tumor`)               | 97.88 | 97.58 | 99.91 | See inside `.zip` file                                        | [Download](https://github.com/ruhyadi/bc-mmsegmentation/releases/download/v1.0/unet-s5-d16_deeplabv3_breast-cancer_2024-07-10.zip) |

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

## Dataset Preparation

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
python tools/dmid/prepare_dataset.py \
    --images_dir data/DMID/tiff_images \
    --anns_dir data/DMID/ground_truth \
    --metadata_csv_path data/DMID/metadata/metadata.csv \
    --output_dir data/DMID/training \
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

## Training

Training segmentation models with MMSegmentation is straightforward. You only need to do the following steps:

#### 1. Create dataset module

Create a dataset module in `mmseg/datasets` directory. In this project, we have created the `breast_cancer.py` dataset module. You can see the details in the [mmseg/datasets/breast_cancer.py](mmseg/datasets/breast_cancer.py).

#### 2. Create dataset configuration

Create a dataset configuration in `configs/_base_/datasets` directory. In this project, we have created the `breast_cancer.py` dataset configuration. You can see the details in the [configs/_base_/datasets/breast_cancer.py](configs/_base_/datasets/breast-cancer.py).

#### 3. Create model configuration

Create a model configuration in `configs/_base_/models` directory. In this project, we have created:

- `unet-s5-d16_deeplabv3_breast-cancer.py`: UNET with DeeplabV3 head. You can see the details in the [configs/unet/unet-s5-d16_deeplabv3_breast-cancer.py](configs/unet/unet-s5-d16_deeplabv3_breast-cancer.py).

### UNET DeeplabV3

UNET with DeeplabV3 head is a model that combines the UNET architecture with the DeeplabV3 head. The model config is defined in the [configs/unet/unet-s5-d16_deeplabv3_breast-cancer.py](configs/unet/unet-s5-d16_deeplabv3_breast-cancer.py) file. The model is trained using the following command:

```bash
python tools/train.py \
    configs/unet/unet-s5-d16_deeplabv3_breast-cancer.py
```

The training script will save the model checkpoints in the `work_dirs/unet-s5-d16_deeplabv3_breast-cancer` directory. The directory will contain the following files:

```bash
work_dirs
└── unet-s5-d16_deeplabv3_breast-cancer
    ├── 20240928_223152 # timestamp
    │   ├── vis_data
    │   │   └── config.py # training configuration
    │   └── 20240928_223048.log # log file
    ├── iter_4000.pth # model checkpoint
    ├── iter_8000.pth # model checkpoint
    ├── ... # model checkpoints
    ├── iter_40000.pth # model checkpoint
    ├── last_checkpoint # last model checkpoint
    └── unet-s5-d16_deeplabv3_breast-cancer.py # model configuration
```

## Evaluation

You can evaluate the model using the following command:

```bash
python tools/test.py \
    configs/unet/unet-s5-d16_deeplabv3_breast-cancer.py \
    work_dirs/unet-s5-d16_deeplabv3_breast-cancer/iter_XXXXX.pth
```

The evaluation script will save the evaluation results in the `work_dirs/unet-s5-d16_deeplabv3_breast-cancer` directory. The directory will contain the following files:

```bash
work_dirs
└── unet-s5-d16_deeplabv3_breast-cancer
    ├── 20240928_223152
    │   ├── vis_data
    │   │   └── config.py
    │   ├── 20240928_223048.json # evaluation results
    │   └── 20240928_223048.log
    ...
```

Inside the `YYYYMMDD_HHMMSS.json` file, you will find the evaluation results in the following format:

```json
{
    "aAcc": 99.94, 
    "mDice": 98.58, 
    "mAcc": 98.4, 
    "data_time": 0.04383463859558105, 
    "time": 0.41584926538689193
}
```

Inside the `YYYYMMDD_HHMMSS.log` file, you will find the evaluation log in the following format:

```bash
...
2024/09/28 22:33:26 - mmengine - INFO - per class results:
2024/09/28 22:33:26 - mmengine - INFO - 
+------------+-------+-------+
|   Class    |  Dice |  Acc  |
+------------+-------+-------+
| background | 99.97 | 99.97 |
|   benign   | 97.64 | 97.16 |
| malignant  | 98.13 | 98.06 |
+------------+-------+-------+
2024/09/28 22:33:26 - mmengine - INFO - Iter(test) [215/215]    aAcc: 99.9400  mDice: 98.5800  mAcc: 98.4000  data_time: 0.0438  time: 0.4158
```

### Confusion Matrix

You can generate the confusion matrix using the following command:

```bash
# generate .pkl file for prediction results
python tools/test.py \
    work_dirs/unet-s5-d16_deeplabv3_breast-cancer/unet-s5-d16_deeplabv3_breast-cancer.py \
    work_dirs/unet-s5-d16_deeplabv3_breast-cancer/iter_40000.pth \
    --out tmp/unet-s5-d16_deeplabv3_breast-cancer/pred_result.pkl

# generate confusion matrix
python tools/analysis_tools/confusion_matrix.py \
    work_dirs/unet-s5-d16_deeplabv3_breast-cancer/unet-s5-d16_deeplabv3_breast-cancer.py \
    tmp/unet-s5-d16_deeplabv3_breast-cancer/pred_result.pkl \
    tmp/unet-s5-d16_deeplabv3_breast-cancer/confusion_matrix \
    --title "UNet-S5-D16-DeepLabV3 Breast Cancer" \
    --rm_background
```

The confusion matrix will be saved in the `tmp/unet-s5-d16_deeplabv3_breast-cancer/confusion_matrix` directory. 

## Prediction

You can predict the image to get the segmentation mask using the following command:

```bash
python tools/dmid/predict.py \
    --config_path work_dirs/unet-s5-d16_deeplabv3_breast-cancer/unet-s5-d16_deeplabv3_breast-cancer.py \
    --weights_path work_dirs/unet-s5-d16_deeplabv3_breast-cancer/iter_40000.pth \
    --images_dir data/DMID/training/images/val \
    --output_dir tmp/predictions
```

You can also add `--metadata_csv_path` to compare the prediction with the ground truth data.

```bash
python tools/dmid/predict.py \
    --config_path work_dirs/unet-s5-d16_deeplabv3_breast-cancer/unet-s5-d16_deeplabv3_breast-cancer.py \
    --weights_path work_dirs/unet-s5-d16_deeplabv3_breast-cancer/iter_40000.pth \
    --images_dir data/DMID/training/images/val \
    --metadata_csv_path data/DMID/metadata/metadata.csv \
    --output_dir tmp/predictions
```

The prediction results will be saved in the `tmp/predictions` directory. The directory will contain the following files:

```bash
tmp
└── predictions
    ├── vis
    │   ├── IMG402.jpg
    │   ├── IMG403.jpg
    │   ├── ...
    │   └── IMG502.jpg
    ├── biner
    │   ├── IMG402.png
    │   ├── IMG403.png
    │   ├── ...
    │   └── IMG502.png
```

## Convert to ONNX

ONNX is an open format built to represent machine learning models. With ONNX format we can convert the model to other formats such as TensorRT, CoreML, and OpenVINO. One of the reasons to convert the model to ONNX in this case is to visualize the model using [Netron](https://netron.app/) tool.

Before converting the model to ONNX, you need to install the `mmdeploy` packages:

```bash
git clone https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy

pip install mmsegmentation mmdeploy onnxruntime
```

Next, you can convert the model to ONNX using the following command:

```bash
python tools/deploy.py \
    configs/mmseg/segmentation_onnxruntime_dynamic.py \
    ../work_dirs/unet-s5-d16_deeplabv3_breast-cancer/unet-s5-d16_deeplabv3_breast-cancer.py \
    ../work_dirs/unet-s5-d16_deeplabv3_breast-cancer/iter_40000.pth \
    demo/resources/cityscapes.png \
    --work-dir mmdeploy_models/mmseg/ort \
    --device cpu \
    --show \
    --dump-info
```

The ONNX model will be saved in the `mmdeploy_models/mmseg/ort` directory.

## Acknowledgement

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab Semantic Segmentation Toolbox and Benchmark.