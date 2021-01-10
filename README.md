# Thesis Semantic Image Segmentation on Satellite Imagery using UNets

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

## Authors

* [Philip Rinkwitz](https://github.com/rinkwitz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
The helper methods in ```Utils/SolarisHelpers.py``` are taken from the [Solaris](https://github.com/CosmiQ/solaris) project
that is licensed under the [Apache-2.0 License](https://github.com/CosmiQ/solaris/blob/master/LICENSE.txt). 
Please refer to that license also if you plan on reusing that code.

[comment]: <> (<p align="center">)

[comment]: <> (<img src="/img/rml_country.jpg" alt="rml country" width="600">)

[comment]: <> (<img src="/img/table_country.png" alt="table country" width="415">)

[comment]: <> (</p>)


[comment]: <> (## Acknowledgements:)

[comment]: <> (The formulas of this README were create using:)

[comment]: <> (* [Codecogs online Latex editor]&#40;https://www.codecogs.com/latex/eqneditor.php&#41;)
