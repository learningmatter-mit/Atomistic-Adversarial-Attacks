# Atomistic Adversarial Attacks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5114385.svg)](https://doi.org/10.5281/zenodo.5114385)

Code for performing adversarial attacks on atomistic systems using NN potentials. The software was based on the paper ["Differentiable sampling of molecular geometries with uncertainty-based adversarial attacks"](https://doi.org/10.1038/s41467-021-25342-8), and implemented by Daniel Schwalbe-Koda and Aik Rui Tan.

The folder [`examples`](examples/) contains several Jupyter notebooks that illustrate the examples shown in the manuscript:

 - [1D double well potential](examples/1D_DoubleWell.ipynb)
 - [2D double well potential](examples/2D_DoubleWell.ipynb)
 - [Performing adversarial attacks on ammonia](examples/Ammonia_attack.ipynb)
 - [Performing adversarial attacks on alanine dipeptide](examples/alanine_attack.py)
 - [Performing adversarial attacks on zeolites](examples/Zeolite_attack.ipynb)
 - [Performing NNMD simulations on ammonia](examples/Ammonia_MD.ipynb)
 - [Performing NNMD simulations on zeolites](examples/Zeolite_MD.ipynb)
 - [Adversarial attack on ANI force field](examples/TorchANI.ipynb)

The folder [`data`](data/) contains three datasets used in the paper: the DFT energies/forces of [ammonia](data/ammonia.pth.tar), OPLS energies/forces of [alanine dipeptide](data/alanine_dipeptide.pth.tar), and [zeolites occluded with neutral molecules](data/zeolite.pth.tar), in the format readable by the [Neural Force Field repo](https://github.com/learningmatter-mit/NeuralForceField).

The full atomistic data is available at the Materials Cloud Archive on the link <https://doi.org/10.24435/materialscloud:2w-6h>.

## Installation from source

This software was tested with [PyTorch 1.4](http://pytorch.org). The installation time highly depends on your internet connection and availability of a `conda` installation, but should not take more than an hour.

We recommend creating a `conda` environment to run the code. To do that, follow the setup instructions at the [Neural Force Field repository](https://github.com/learningmatter-mit/NeuralForceField).

```bash
conda upgrade conda
conda create -n nff python=3.7 scikit-learn pytorch=1.4.0 cudatoolkit=10.0 ase pandas pymatgen sympy rdkit hyperopt jq openbabel -c pytorch -c conda-forge -c rdkit -c openbabel
```

Then, install the remaining requirements using `pip`:

```bash
conda activate nff
pip install ipykernel nglview sigopt e3fp
```

To ensure that the `nff` environment is accessible through Jupyter, add the the `nff` display name:

```bash
python -m ipykernel install --user --name nff --display-name "nff"
```

## Tutorials on how to use the NN potential

More tutorials are available on the [Neural Force Field repository](https://github.com/learningmatter-mit/NeuralForceField)

## Citing

The reference for the paper is the following:

```
@article{schwalbe2021differentiable,
  title={Differentiable sampling of molecular geometries with uncertainty-based adversarial attacks},
  author={Schwalbe-Koda, Daniel and Tan, Aik Rui and G{\'o}mez-Bombarelli, Rafael},
  journal={Nature Communications},
  volume={12},
  pages={5104},
  year={2021},
  publisher={Nature Publishing Group}
}
```

