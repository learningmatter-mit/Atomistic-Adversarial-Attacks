from nff.train.hooks import Hook


class RequiresGradHook(Hook):
    def on_batch_begin(self, trainer, batch):
        batch[0].requires_grad = True
        return batch
