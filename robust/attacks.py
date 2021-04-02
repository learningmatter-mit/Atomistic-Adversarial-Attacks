import torch as ch
from robust.train import batch_to, batch_detach
from robust.data import PotentialDataset


DEFAULT_EPS = 1e-2
MAX_EPOCHS = 100


class Attack:
    def __init__(
        self,
        optim_cls=ch.optim.Adam,
        optim_kws={},
        delta_std=0,
        **kwargs,
    ):
        self.optim_kws = optim_kws
        self.optim_cls = optim_cls
        self.delta_std = delta_std

    def get_optim(self, delta):
        return self.optim_cls([delta], **self.optim_kws)

    def create_delta(self, x):
        delta = (
            ch.zeros_like(x, device=x.device)
            + self.delta_std * ch.randn_like(x, device=x.device)
        )
        delta.requires_grad = True
        optim = self.get_optim(delta)

        return delta, optim

    def attack(self, x, delta):
        raise NotImplementedError


class SumAttack(Attack):
    def attack(self, x, delta):
        return x + delta


class AdversarialAttacker:
    def __init__(
        self,
        model,
        loss_fn,
        loader,
        attack,
        epsilon=DEFAULT_EPS,
        **kwargs,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.loader = loader
        self.attack = attack
        self.eps = epsilon

    def to(self, device):
        """Changes the device"""
        self.model.device = device
        self.model.to(device)

    def train(self, device, n_epochs=MAX_EPOCHS):
        self.to(device)

        results = PotentialDataset.from_empty_dataset()
        for batch in self.loader:
            batch = batch_to(batch, device)

            attacked = self.attack_batch(batch, device, n_epochs)
            attacked = batch_detach(attacked)
            results.add_batch(attacked)

        return results

    def attack_batch(self, batch, device, n_epochs):
        x_init = batch[0]
        delta, optim = self.attack.create_delta(x_init)

        for epoch in range(n_epochs):
            optim.zero_grad()

            x = self.attack.attack(x_init, delta)
            results = self.model(x)

            loss = self.loss_fn(batch, results)
            loss.backward()
            optim.step()

        return self.model(x)


class RandomAttacker(AdversarialAttacker):
    def attack_batch(self, batch, device, n_epochs):
        x_init = batch[0]
        delta = self.eps * (2 * ch.rand_like(x_init, device=x_init.device) - 1)
        x = self.attack.attack(x_init, delta)
        x.requires_grad = True

        return self.model(x)
