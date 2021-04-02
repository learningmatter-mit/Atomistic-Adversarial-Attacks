import torch as ch
from nff.train.metrics import MeanAbsoluteError


class MAE(MeanAbsoluteError):
    @staticmethod
    def loss_fn(y, yp):
        diff = y[..., None].expand_as(yp) - yp
        return ch.sum(ch.abs(diff).view(-1)).detach().cpu().data.numpy()
