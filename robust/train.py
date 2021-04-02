import os
import numpy as np
import torch as ch

from robust.data import PotentialDataset


MAX_EPOCHS = 1000


def batch_to(batch, device):
    return [x.to(device) for x in batch]


def batch_detach(batch):
    return [b.detach().cpu() if hasattr(b, "detach") else b for b in batch]


class Trainer:
    r"""Class to train a model.

    This contains an internal training loop which takes care of validation
        and can be extended with custom functionality using hooks.

    Args:
       model_path (str): path to the model directory.
       model (torch.Module): model to be trained.
       loss_fn (callable): training loss function.
       optimizer (torch.optim.optimizer.Optimizer): training optimizer.
       train_loader (torch.utils.data.DataLoader): data loader for
         training set.
       validation_loader (torch.utils.data.DataLoader): data loader for
         validation set.
       checkpoints_to_keep (int, optional): number of saved checkpoints.
       checkpoint_interval (int, optional): intervals after which checkpoints
         is saved.
       hooks (list, optional): hooks to customize training process.

    """

    def __init__(
        self,
        model_path,
        model,
        loss_fn,
        optimizer,
        train_loader,
        validation_loader,
        mini_batches=1,
        checkpoints_to_keep=1,
        checkpoint_interval=10,
        validation_interval=1,
        hooks=[],
    ):
        self.model_path = model_path
        self.checkpoint_path = os.path.join(self.model_path, "checkpoints")
        self.best_model = os.path.join(self.model_path, "best_model")
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.validation_interval = validation_interval
        self.checkpoints_to_keep = checkpoints_to_keep
        self.hooks = hooks

        self._model = model
        self._stop = False
        self.checkpoint_interval = checkpoint_interval

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if os.path.exists(self.checkpoint_path):
            self.restore_checkpoint()
        else:
            os.makedirs(self.checkpoint_path)
            self.epoch = 0
            self.step = 0
            self.best_loss = float("inf")
            self.store_checkpoint()

    def to(self, device):
        """Changes the device"""
        self._model.device = device
        self._model.to(device)
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def _check_is_parallel(self):
        return True if isinstance(self._model, ch.nn.DataParallel) else False

    def _load_model_state_dict(self, state_dict):
        if self._check_is_parallel():
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)

    def get_best_model(self):
        return ch.load(self.best_model)

    @property
    def state_dict(self):
        state_dict = {
            "epoch": self.epoch,
            "step": self.step,
            "best_loss": self.best_loss,
            "optimizer": self.optimizer.state_dict(),
            "hooks": [h.state_dict for h in self.hooks],
        }
        if self._check_is_parallel():
            state_dict["model"] = self._model.module.state_dict()
        else:
            state_dict["model"] = self._model.state_dict()
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.best_loss = state_dict["best_loss"]
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self._load_model_state_dict(state_dict["model"])

        for h, s in zip(self.hooks, self.state_dict["hooks"]):
            h.state_dict = s

    def store_checkpoint(self):
        chkpt = os.path.join(
            self.checkpoint_path, "checkpoint-" + str(self.epoch) + ".pth.tar"
        )
        ch.save(self.state_dict, chkpt)

        chpts = [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pth.tar")]
        if len(chpts) > self.checkpoints_to_keep:
            chpt_epochs = [int(f.split(".")[0].split("-")[-1]) for f in chpts]
            sidx = np.argsort(chpt_epochs)
            for i in sidx[: -self.checkpoints_to_keep]:
                os.remove(os.path.join(self.checkpoint_path, chpts[i]))

    def restore_checkpoint(self, epoch=None):
        if epoch is None:
            epoch = max(
                [
                    int(f.split(".")[0].split("-")[-1])
                    for f in os.listdir(self.checkpoint_path)
                    if f.startswith("checkpoint")
                ]
            )

        chkpt = os.path.join(
            self.checkpoint_path, "checkpoint-" + str(epoch) + ".pth.tar"
        )
        self.state_dict = ch.load(chkpt)

    def train(self, device, n_epochs=MAX_EPOCHS):
        """Train the model for the given number of epochs on a specified device.

        Args:
            device (torch.Device): device on which training takes place.
            n_epochs (int): number of training epochs.

        Note: Depending on the `hooks`, training can stop earlier than `n_epochs`.

        """
        self.to(device)

        self._stop = False

        loss = ch.tensor(0.0).to(device)
        self.optimizer.zero_grad()

        for h in self.hooks:
            h.on_train_begin(self)

        try:
            for _ in range(n_epochs):
                self._model.train()

                self.epoch += 1

                for h in self.hooks:
                    h.on_epoch_begin(self)

                if self._stop:
                    break

                for j, batch in enumerate(self.train_loader):
                    batch = batch_to(batch, device)

                    for h in self.hooks:
                        h.on_batch_begin(self, batch)

                    results = self._model(batch[0])
                    loss += self.loss_fn(batch, results)
                    self.step += 1

                    loss.backward()
                    self.optimizer.step()

                    for h in self.hooks:
                        h.on_batch_end(self, batch, results, loss)

                    loss = ch.tensor(0.0).to(device)
                    self.optimizer.zero_grad()

                    if self._stop:
                        break

                if self.epoch % self.checkpoint_interval == 0:
                    self.store_checkpoint()

                # validation
                if self.epoch % self.validation_interval == 0 or self._stop:
                    self.validate(device)

                for h in self.hooks:
                    h.on_epoch_end(self)

                if self._stop:
                    break

            # Training Ends
            # run hooks & store checkpoint
            for h in self.hooks:
                h.on_train_ends(self)

            self.store_checkpoint()

        except Exception as e:
            for h in self.hooks:
                h.on_train_failed(self)

            raise e

    def validate(self, device):
        """Validate the current state of the model using the validation set"""

        self._model.eval()

        for h in self.hooks:
            h.on_validation_begin(self)

        val_loss = 0.0
        n_batches = 0

        for val_batch in self.validation_loader:
            val_batch = batch_to(val_batch, device)

            for h in self.hooks:
                h.on_validation_batch_begin(self)

            x = val_batch[0]
            x.requires_grad = True

            results = self._model(x)

            val_loss += self.loss_fn(val_batch, results).data.cpu().numpy()
            n_batches += 1

            for h in self.hooks:
                h.on_validation_batch_end(self, val_batch, results)

        # weighted average over batches
        val_loss /= n_batches

        if self.best_loss > val_loss:
            self.best_loss = val_loss
            ch.save(self._model, self.best_model)

        for h in self.hooks:
            h.on_validation_end(self, val_loss)

    def evaluate(self, loader, device):
        return evaluate(self.get_best_model(), loader, self.loss_fn, device)


def evaluate(
    model,
    loader,
    loss_fn,
    device,
):
    model.eval()
    model.to(device)

    eval_loss = 0.0
    n_batches = 0

    all_results = PotentialDataset.from_empty_dataset()
    all_batches = PotentialDataset.from_empty_dataset()

    for batch in loader:
        # append batch_size
        batch = batch_to(batch, device)

        x = batch[0]
        x.requires_grad = True

        results = model(x)

        eval_batch_loss = loss_fn(batch, results).data.cpu().numpy()

        eval_loss += eval_batch_loss
        n_batches += 1

        all_results.add_batch(batch_detach(results))
        all_batches.add_batch(batch_detach(batch))

    eval_loss /= n_batches

    return all_results, all_batches, eval_loss
