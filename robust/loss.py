import torch as ch
from robust.data import PotentialDataset


class LossFunction:
    def __init__(self, energy_coef=0.1, forces_coef=1, **kwargs):
        self.e_coef = energy_coef
        self.f_coef = forces_coef

    def __call__(self, batch, pred, normalize=False):
        # pred = (x, energy, forces)
        pred_energy = pred[1]
        pred_forces = pred[2]

        # batch = (x, energy, forces)
        targ_energy = batch[1][..., None].expand_as(pred_energy)
        targ_forces = batch[2][..., None].expand_as(pred_forces)

        energy_loss = self.loss_fn(pred_energy, targ_energy)
        forces_loss = self.loss_fn(pred_forces, targ_forces)

        return self.e_coef * energy_loss + self.f_coef * forces_loss

    def loss_fn(self, pred, targ):
        raise NotImplementedError


class MeanSquareLoss(LossFunction):
    def loss_fn(self, pred, targ):
        return ((pred - targ) ** 2).mean()


class AdvLoss(LossFunction):
    def __init__(
        self,
        train: PotentialDataset,
        temperature: float = 1,
        **kwargs
    ):
        self.e = train.e
        self.temperature = temperature

    def boltzmann_probability(self, e):
        return ch.exp(-e / self.temperature)

    @property
    def partition_fn(self):
        return self.boltzmann_probability(self.e).mean()

    def probability_fn(self, yp):
        return self.boltzmann_probability(yp) / self.partition_fn

    def loss_fn(self, x, e, f=None, s=None):
        if s is not None:
            return -s.var(-1).mean() * self.probability_fn(e.mean(-1).reshape(-1, 1))
        if f is not None:
            return -f.var(-1).mean(-1, keepdims=True) * self.probability_fn(e.mean(-1).reshape(-1, 1))

    def __call__(self, batch, results, lattice=False):
        if lattice:
            x, e, f, s = results
            return self.loss_fn(x, e, s).sum()
        else:
            x, e, f = results
            return self.loss_fn(x, e, f).sum()


class AdvLossEnergyUncertainty(AdvLoss):
    def loss_fn(self, x, e, f):
        return -e.var(-1) * self.probability_fn(e.mean(-1))# * self.rmsd.loss_fn(x)


class RmsdLoss(LossFunction):
    def __init__(
        self,
        train: PotentialDataset,
        **kwargs,
    ):
        """Calculates the score of a point according to its RMSD
        with respect to the training data"""
        self.x = train.x

    def __call__(self, batch, results):
        x, e, f = results
        return self.loss_fn(x).sum()

    def loss_fn(self, x):
        distances = (self.x[:, None, ...].to(x.device) - x[None, ...]).norm(dim=-1)
        vals, idx = distances.min(dim=0)
        return vals
