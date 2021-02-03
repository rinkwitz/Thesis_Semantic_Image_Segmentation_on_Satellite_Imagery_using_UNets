# Thesis Semantic Image Segmentation on Satellite Imagery using UNets
This project is part of the master thesis *Possibilities to improve the performance and 
robustness of U-Nets for image segmentation in satellite images with a focus on attention 
mechanisms and transfer learning* with the 
[Department of Data Science and Knowledge Engineering](https://www.maastrichtuniversity.nl/research/department-data-science-and-knowledge-engineering-dke) 
at Maastricht University.
* [Thesis](https://drive.protonmail.com/urls/3F74H6B12W#hx7hdO81NAG8)
* [Presentation](https://drive.protonmail.com/urls/HY6957FBNW#Su3FtBkOnOQQ)

## Abstract

This master thesis aims to perform semantic segmentation of buildings on satellite images from
the SpaceNet challenge 1 dataset using the U-Net architecture.
To enable a more focused use of feature-maps, soft attention mechanisms
are integrated into the U-Net and examined.
Furthermore, possibilities of transfer learning are investigated by
using convolutional neural networks pre-trained
on ImageNet as encoders for the U-Net.
Finally, the performance and robustness for the segmentation task is evaluated for both approaches.

## Prerequisites

It is assumed that you have [anaconda Python](https://www.anaconda.com/) installed. You can create a Python
environment with the required dependencies by following:

```
conda create -n thesis_env python=3.7
conda activate thesis_env
conda install numpy pandas rasterio shapely geopandas matplotlib imageio scikit-image
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install tifffile
```

If your CUDA versions differ please refer to [Start Locally | PyTorch](https://pytorch.org/get-started/locally/).

## Model weights

The model weights can be found at:

* [U-Net](https://drive.protonmail.com/urls/0K1DQWA7BW#FHqatMWKW81I)
* [Attention U-Net](https://drive.protonmail.com/urls/H0M2CM57ZR#omG2xLYB4R9R)
* [CBAM U-Net](https://drive.protonmail.com/urls/KQW02V5HTW#1re9Edplo6QY)
* [Residual Attention U-Net](https://drive.protonmail.com/urls/8PVP5JE2J4#JYg7IsbbOXK7)
* [scAG U-Net](https://drive.protonmail.com/urls/1P81WARY6C#Z51ijrV9bdg0)
* [DenseNet121 U-Net](https://drive.protonmail.com/urls/D5QT4C92YW#3DE25wzni9Ci)
* [MobileNetV2 U-Net](https://drive.protonmail.com/urls/TJW6F5THT8#MtqNAn0k1YxZ)
* [ResNet34 U-Net](https://drive.protonmail.com/urls/6T6354SJZC#EP8B6RmRojVA)
* [VGG11 U-Net](https://drive.protonmail.com/urls/8K3F124Y9C#dc7c0LcjCfri)

## Authors

* [Philip Rinkwitz](https://github.com/rinkwitz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
The helper methods in ```Utils/SolarisHelpers.py``` are taken from the [Solaris](https://github.com/CosmiQ/solaris) project
that is licensed under the [Apache-2.0 License](https://github.com/CosmiQ/solaris/blob/master/LICENSE.txt). 
Please refer to that license if you plan on reusing that code.
