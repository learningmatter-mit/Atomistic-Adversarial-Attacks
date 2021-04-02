# Atomistic Adversarial Attacks

Code for performing adversarial attacks on atomistic systems using NN potentials. The software was based on the paper ["Differentiable sampling of molecular geometries with uncertainty-based adversarial attacks"](https://arxiv.org/abs/2101.11588), and implemented by Daniel Schwalbe-Koda and Aik Rui Tan.

The folder [`examples`](examples/) contains two Jupyter notebooks that illustrate the examples shown in the manuscript:

 - [1D double well potential](examples/1D_DoubleWell.ipynb)
 - [2D double well potential](examples/2D_DoubleWell.ipynb)

More examples regarding the use of atomistic systems will be added soon.

## Installation from source

This software was tested with [PyTorch 1.4](http://pytorch.org)

We recommend creating a `conda` environment to run the code. To do that, use the following commands:

```bash
conda upgrade conda
conda create -n advsampling python=3.7 scipy numpy pytorch=1.4 jupyter -c pytorch
```

Then, install the remaining requirements using `pip`:

```bash
conda activate advsampling
pip install ipykernel
```

To ensure that the `advsampling` environment is accessible through Jupyter, add the the `advsampling` display name:

```bash
python -m ipykernel install --user --name advsampling --display-name "advsampling"
```

## Citing

The reference for the paper is the following:

```
@article{schwalbe2021differentiable,
  title={Differentiable sampling of molecular geometries with uncertainty-based adversarial attacks},
  author={Schwalbe-Koda, Daniel and Tan, Aik Rui and G{\'o}mez-Bombarelli, Rafael},
  journal={arXiv:2101.11588},
  year={2021}
}
```

