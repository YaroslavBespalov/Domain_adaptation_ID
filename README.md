# BRULE

Implementation of StyleGAN2, CycleGAN in PyTorch to domain adaptation task.

## Main requirements

* PyTorch 1.5.0
* CUDA 10.0+
* GPU 16Gb

## Usage

**Download datasets:**
CALGARY-CAMPINAS PUBLIC BRAIN MR DATASET (https://sites.google.com/view/calgary-campinas-dataset/home),
Extract them into some `<data_folder>`.
Set the required paths to the `<data_folder>` in the config file `./parameters/path.py` (DGXPath.ausland).


Install python requirements:

> pip3 install -r requirements.txt

**Compile cuda extensions of stylegan2:**

> python3 ./gans_pytorch/gan/nn/stylegan/op/setup.py install

Start tensorboard:

> tensorboard --logdir=<logs_dir>

set correspondent path to the `<logs_dir>` in the config file `./parameters/path.py` (DGXPath.homa).

**Train CycleGAN model** on Calgary dataset:

> cd src/examples

> python3 cycle_gan.py

**Train StyleGAN2 model** on Calgary dataset:

> cd src/examples

> python3 DA_stylegan2.py


## Project structure

- `src/dataset/lazy_loader.py` pytorch dataloders
- `src/examples` train and test scripts
- `gans` library with GAN models and utils for min-max optimization
- `src/loss` regularizers and the ther loss components
- `src/parameters` loss coefficients for GAN and encoder training
