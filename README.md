# SSHN: Sparse and Structured Hopfield Networks
Official implementation of paper **Sparse and Structured Hopfield Networks**.

*Saul Santos*, *Vlad Niculae*, *Daniel McNamee* and *André Martins*

**Abstract**: *Modern Hopfield networks have enjoyed recent interest due to their connection to attention in transformers. Our paper provides a unified framework for sparse Hopfield networks by establishing a link with Fenchel-Young losses. The result is a new family of Hopfield-Fenchel-Young energies whose update rules are end-to-end differentiable sparse transformations. We reveal a connection between loss margins, sparsity, and exact memory retrieval. We further extend this framework to structured Hopfield networks via the SparseMAP transformation, which can retrieve pattern associations instead of a single pattern. Experiments on multiple instance learning and text rationalization demonstrate the usefulness of our approach.*

----------

**If you use this code in your work, please cite our paper.**

----------

## Resources

- [Paper](http://arxiv.org/abs/2402.13725) (arXiv)

All material is made available under the MIT license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.


## Synthetic, MNIST and Multiple Instance Learning Experiments
### Python requirements and installation

This code was tested on `Python 3.10.10`. To install, follow these steps:

1. In a virtual environment, first install Cython: `pip install cython`
2. Clone the [Eigen](https://gitlab.com/libeigen/eigen.git) repository to the main folder: `git clone git@gitlab.com:libeigen/eigen.git`
3. Clone the [LP-SparseMAP](https://github.com/nunonmg/lp-sparsemap) fork repository to main folder, and follow the installation instructions found there
4. Install the requirements: `pip install -r requirements.txt`
5. Run the corresponding scripts

### Reproducibility
#### MNIST MIL
Run the script `MNIST_bags.py` with the desired parameters (nomenclature can be found in the beginning of the script)

#### Benchmarks MIL

Download and upzip the dataset

```bash
$ wget http://www.cs.columbia.edu/~andrews/mil/data/MIL-Data-2002-Musk-Corel-Trec9-MATLAB.tgz 
```

Run the script `MIL_Data_2002.py` with the desired parameters (nomenclature can be found in the beginning of the script)

#### Countours and Basins of Attraction
Run the scripts `countours.py` and `basins.py` 

#### Metastable State Counting
Run the script `MNIST_metastable.py`

## Spectra Experiments
### Python requirements and installation
Follow the instructions of the branch in [hopfield-spectra](https://github.com/deep-spin/spectra-rationalization/tree/hopfield-spectra)

## Acknowledgment

The experiments in this work benefit from the following open-source codes:
* Ramsauer, Hubert, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber et al. "Hopfield networks is all you need." arXiv preprint arXiv:2008.02217 (2020). https://github.com/ml-jku/hopfield-layers
* Martins, Andre, and Ramon Astudillo. "From softmax to sparsemax: A sparse model of attention and multi-label classification." In International conference on machine learning, pp. 1614-1623. PMLR, 2016. https://github.com/deep-spin/entmax
* Correia, Gonçalo M., Vlad Niculae, and André FT Martins. "Adaptively sparse transformers." arXiv preprint arXiv:1909.00015 (2019). https://github.com/deep-spin/entmax
* Peters, Ben; Niculae, Vlad; Martins, André FT. "Sparse Sequence-to-Sequence Models." In Proceedings of ACL, 2019. [Online] Available: https://www.aclweb.org/anthology/P19-1146.  https://github.com/deep-spin/entmax
* Guerreiro, N. M. and Martins, A. F. T. Spectra: Sparse structured text rationalization. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 6534–6550, 2021. https://github.com/deep-spin/spectra-rationalization/tree/hopfield-spectra
* Ilse, Maximilian, Jakub Tomczak, and Max Welling. "Attention-based deep multiple instance learning." In International conference on machine learning, pp. 2127-2136. PMLR, 2018. https://github.com/AMLab-Amsterdam/AttentionDeepMIL

