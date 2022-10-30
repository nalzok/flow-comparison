# Generative Latent Flow

A pytorch implementation of "Generative Latent Flow".

### Prerequisites

- `numpy`, `pytorch`, `tqdm`.

- `FrEIA`, which is the package for flow model. It can be installed by running:  

  `pip install git+https://github.com/VLL-HD/FrEIA.git`.
  
  

###Training

- `python train_glf.py` will train a GLF model for MNIST.

- `python train_vaeflow.py` will train GLF+flow prior model for MNIST.

- `python generate_sample.py` will generate samples from given model. You will need to change the name to the model you want to use in `load_check_point` function.

### Compute FID score

- A folder of real test images can be obtained in various ways.
- Once you have two folders of images (one for real images and one for samples), you may
  1. compute FID scores using their original implementation at https://github.com/bioinf-jku/TTUR
  2. compute precision and recall using their original implementation at https://github.com/msmsajjadi/precision-recall-distributions