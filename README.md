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

### Training

```bash
python tools/train.py \
    configs/unet/unet-s5-d16_deeplabv3_breast-cancer.py
```

### Testing

```bash
python tools/test.py \
    configs/unet/unet-s5-d16_deeplabv3_breast-cancer.py \
    work_dirs/unet-s5-d16_deeplabv3_breast-cancer_640x640_best/iter_40000.pth
```

## Acknowledgement

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab Semantic Segmentation Toolbox and Benchmark.