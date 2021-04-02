import torch as ch
from torch.optim import Adam
from torch.utils.data import DataLoader

from nff.train import hooks as nff_hooks

from robust.models import NnEnsemble, NnRegressor
from robust import metrics, hooks, train, loss, attacks, data


DEFAULT_NAME = "train"
BATCH_SIZE = 50
MAX_EPOCHS = 300


class ForwardPipeline:
    def __init__(
        self,
        dset,
        model_params,
        loss_params,
        optim_params,
        train_params,
        attack_params,
        name=DEFAULT_NAME,
        dset_train=None,
        dset_train_weight=4,
    ):
        self.name = name
        self.dset = dset

        self.dset_train_weight = dset_train_weight
        if dset_train is None:
            self.dset_train = data.PotentialDataset.from_empty_dataset()
        else:
            self.dset_train = dset_train

        self.model_params = model_params
        self.loss_params = loss_params
        self.optim_params = optim_params
        self.train_params = train_params
        self.attack_params = attack_params

        self.train_loader, self.val_loader, self.test_loader = self.get_loaders(
            train_params["batch_size"]
        )
        self.trainer = self.get_trainer()

    def copy(self, new_name=None):
        if new_name is None:
            new_name = self.name

        return self.__class__(
            self.dset,
            self.model_params.copy(),
            self.loss_params.copy(),
            self.optim_params.copy(),
            self.train_params.copy(),
            self.attack_params.copy(),
            new_name,
        )

    def augment_train_set(self, train):
        newtrain = train
        for i in range(self.dset_train_weight):
            newtrain += self.dset_train.copy()
        return newtrain

    def get_loaders(self, batch_size):
        train, val, test = self.dset.split_train_validation_test()

        train_loader = DataLoader(
            self.augment_train_set(train),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    def create_model(self):
        num_nets = self.model_params["num_networks"]
        model = NnEnsemble([NnRegressor(**self.model_params) for _ in range(num_nets)])

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(trainable_params, **self.optim_params)

        return model, optimizer

    def get_loss_fn(self):
        return loss.MeanSquareLoss(**self.loss_params)

    def get_adv_loss(self):
        loss_type = self.attack_params.get('uncertainty', 'forces')

        if loss_type == 'energy':
            return loss.AdvLossEnergyUncertainty(self.dset + self.dset_train, **self.loss_params)

        return loss.AdvLoss(self.dset + self.dset_train, **self.loss_params)

    def get_metrics(self):
        return [metrics.MAE(1, name="MAE Energy"), metrics.MAE(2, name="MAE Forces")]

    def get_hooks(self, optimizer):
        return [
            nff_hooks.MaxEpochHook(self.train_params.get("max_epochs", MAX_EPOCHS)),
            nff_hooks.PrintingHook(
                self.name,
                metrics=self.get_metrics(),
                separator=" | ",
                time_strf="%M:%S",
                every_n_epochs=25,
            ),
            nff_hooks.ReduceLROnPlateauHook(
                optimizer=optimizer,
                patience=50,
                factor=0.5,
                min_lr=1e-7,
                window_length=1,
                stop_after_min=True,
            ),
            hooks.RequiresGradHook(),
        ]

    def get_trainer(self):
        model, optimizer = self.create_model()

        T = train.Trainer(
            model_path=self.name,
            model=model,
            loss_fn=self.get_loss_fn(),
            optimizer=optimizer,
            train_loader=self.train_loader,
            validation_loader=self.val_loader,
            checkpoint_interval=50,
            hooks=self.get_hooks(optimizer),
        )

        return T

    def train(self, device, n_epochs):
        self.trainer.train(device, n_epochs)
        return self.trainer.get_best_model()

    def evaluate(self, loader, device):
        return self.trainer.evaluate(loader, device)

    def get_attacker(self):
        attack = attacks.SumAttack(
            **self.attack_params,
        )

        if self.attack_params.get('random_attack', False):
            attacker_cls = attacks.RandomAttacker
        else:
            attacker_cls = attacks.AdversarialAttacker

        return attacker_cls(
            self.trainer.get_best_model(),
            self.get_adv_loss(),
            self.train_loader,
            attack,
            **self.attack_params,
        )

    def attack(self, device, n_epochs):
        self.attacker = self.get_attacker()
        return self.attacker.train(device, n_epochs)
