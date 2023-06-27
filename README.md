# Cell Seg
**Krishnaswamy Lab, Yale University**

<!-- [![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/ChenLiu-1996/DiffusionSpectralEntropy.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/DiffusionSpectralEntropy/) -->


This is the **official** implementation of


## Preparation

### Environment
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
**Some packages may no longer be required.**
```
conda create --name cellseg pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda activate cellseg
conda install -c anaconda scikit-image scikit-learn pillow matplotlib seaborn tqdm
python -m pip install opencv-python
```

<!-- python -m pip install tinyimagenet
python -m pip install natsort
python -m pip install phate
python -m pip install DiffusionEMD
python -m pip install magic-impute -->
