# SSHN: Sparse and Structured Hopfield Networks
Official implementation of paper **Sparse and Structured Hopfield Networks**.

*Saul Santos*, *Vlad Niculae*, *Daniel McNamee* and *André Martins*

**Abstract**: *Modern Hopfield networks have enjoyed recent interest due to their connection to attention in transformers. Our paper provides a unified frame- work for sparse Hopfield networks by establishing a link with Fenchel-Young losses. The result is a new family of Hopfield-Fenchel-Young energies whose update rules are end-to-end differentiable sparse transformations. We reveal a connection between loss margins, sparsity, and exact memory retrieval. We further extend this framework to structured Hopfield networks via the SparseMAP transformation, which can retrieve pattern associations instead of a single pattern. Experiments on multiple instance learning and text rationalization demonstrate the usefulness of our approach.*

----------

**If you use this code in your work, please cite our paper.**

----------

## Resources

- [Paper](to add) (arXiv)

All material is made available under the MIT license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.


## Synthetic, MNIST and Multiple Instance Learning Experiments
### Python requirements and installation

This code was tested on `Python 3.10.10`. To install, follow these steps:

1. In a virtual environment, first install Cython: `pip install cython`
2. Clone the [Eigen](https://gitlab.com/libeigen/eigen.git) repository to the main folder: `git clone git@gitlab.com:libeigen/eigen.git`
3. Clone the [LP-SparseMAP](https://github.com/nunonmg/lp-sparsemap) fork repository to main folder, and follow the installation instructions found there
4. Install the requirements: `pip install -r requirements.txt`
5. Install the `spectra-rationalization` package: `pip install .` (or in editable mode if you want to make changes: `pip install -e .`)

---

## Spectra Experiments
