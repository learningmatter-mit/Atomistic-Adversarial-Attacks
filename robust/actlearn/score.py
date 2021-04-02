import torch as ch
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage

from robust.data import PotentialDataset
from robust.loss import RmsdLoss


DEFAULT_PERCENTILE = 80
DEFAULT_SCORES = {
    "UncertaintyPercentile": {"percentile": 80},
    "RmsdScore": {"threshold": 0.1},
}


class Score:
    """Calculates the score of data points according to
    the given training data results.
    """

    def __call__(self, test):
        return [self.calculate_score(*data) for data in test]

    def calculate_score(self, x, e, f):
        raise NotImplementedError


class UncertaintyPercentile(Score):
    def __init__(
        self,
        train: PotentialDataset,
        percentile: int = DEFAULT_PERCENTILE,
    ):
        """Calculates the score of a point according to a threshold
        of uncertainty taken from the training data."""
        self.threshold = self.get_train_threshold(train, percentile)

    def get_train_threshold(self, train, percentile):
        return stats.scoreatpercentile(train.f.var(-1), percentile)

    def __call__(self, test: PotentialDataset):
        score = test.f.var(-1) >= self.threshold
        return score.reshape(len(test), -1).all(-1)

    def calculate_score(self, x, e, f):
        return f.var(-1) > self.threshold


class RmsdScore(Score):
    def __init__(
        self,
        train: PotentialDataset,
        threshold: float,
    ):
        """Calculates the score of a point according to its RMSD
        with respect to the training data"""
        self.rmsd = RmsdLoss(train)
        self.threshold = threshold

    def __call__(self, test: PotentialDataset):
        return self.rmsd.loss_fn(test.x) > self.threshold


class ClusterScore(Score):
    def __init__(
        self,
        train: PotentialDataset,
        threshold: float,
        criterion: str = 'distance',
    ):
        """Performs a hierarchical clustering on the attacked data
        such that points too close to one another are deduplicated"""
        self.threshold = threshold
        self.criterion = criterion

    def __call__(self, test: PotentialDataset):
        x = test.x
        dm = (x[None, ...] - x[:, None, ...]).norm(dim=-1)
        clusters = fcluster(
            linkage(
                squareform((dm + dm.T) / 2)
            ),
            self.threshold,
            criterion=self.criterion
        )
        unique_idx = []
        for c in np.unique(clusters):
            idx = np.where(clusters == c)[0]
            unique_idx.append(idx[0])

        return ch.tensor([i in unique_idx for i in range(len(x))])
