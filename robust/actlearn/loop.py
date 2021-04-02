import os
import torch as ch
import numpy as np

from robust import train, loss, attacks, data
from . import score
from .pipeline import ForwardPipeline


NUM_GENERATIONS = 5
NAME_PREFIX = 'loop_gen'
MAX_SAMPLED_POINTS = 20

SCORE_CLASSES = {
    "UncertaintyPercentile": score.UncertaintyPercentile,
    "RmsdScore": score.RmsdScore,
    "ClusterScore": score.ClusterScore,
}


class ActiveLearning:
    def __init__(
        self,
        pipeline,
        potential,
        num_generations=NUM_GENERATIONS,
        device="cpu",
        train_epochs=train.MAX_EPOCHS,
        attack_epochs=attacks.MAX_EPOCHS,
        scores=score.DEFAULT_SCORES,
        name_prefix=NAME_PREFIX,
        max_sampled_points=MAX_SAMPLED_POINTS,
    ):
        self.pipeline = pipeline
        self.potential = potential
        self.num_generations = num_generations
        self.generations = []

        self.device = device
        self.train_epochs = train_epochs
        self.attack_epochs = attack_epochs

        self.scores = scores
        self.max_samples = max_sampled_points

        self.prefix = name_prefix
        self.init_path()

    def init_path(self):
        if not os.path.exists(self.prefix):
            os.mkdir(self.prefix)

    def train(self):
        return self.pipeline.train(self.device, self.train_epochs)

    def attack(self):
        return self.pipeline.attack(self.device, self.attack_epochs)

    def evaluate(self, results: data.PotentialDataset):
        return data.PotentialDataset(*self.potential(results.x))

    def create_new_dataset(self, x, e, f):
        newx = ch.cat([self.pipeline.dset.x, x])
        newe = ch.cat([self.pipeline.dset.e, e])
        newf = ch.cat([self.pipeline.dset.f, f])

        return self.pipeline.dset.__class__(
            newx,
            newe,
            newf,
        )

    def deduplicate(
        self,
        train_results: data.PotentialDataset,
        attack_results: data.PotentialDataset,
    ):
        scores = [
            SCORE_CLASSES[name](train_results, **kwargs)
            for name, kwargs in self.scores.items()
        ]

        idx = np.bitwise_and.reduce([
            score(attack_results).numpy().reshape(-1)
            for score in scores
        ])

        dset = data.PotentialDataset(*attack_results[idx])

        if len(dset) > self.max_samples:
            return dset.sample(self.max_samples)

        return dset

    def get_generation_name(self, gen):
        return f"{self.prefix}/gen_{gen + 1}"

    def loop(self):
        for gen in range(1, self.num_generations + 1):
            self.log(f"GEN {gen}: training model")
            self.train()

            self.log(f"GEN {gen}: evaluating")
            train_results, _, _ = self.pipeline.evaluate(
                self.pipeline.train_loader,
                self.device,
            )

            self.log(f"GEN {gen}: attacking")
            attack_results = self.attack()
            attack_results = self.deduplicate(train_results, attack_results)

            self.log(f"GEN {gen}: evaluating attack")
            newdata = self.evaluate(attack_results)
            newdata = newdata + self.pipeline.dset_train

            self.generations.append(
                {
                    "generation": gen,
                    "pipeline": self.pipeline,
                    "attacks": attack_results,
                }
            )

            self.pipeline = ForwardPipeline(
                self.pipeline.dset,
                self.pipeline.model_params.copy(),
                self.pipeline.loss_params.copy(),
                self.pipeline.optim_params.copy(),
                self.pipeline.train_params.copy(),
                self.pipeline.attack_params.copy(),
                name=self.get_generation_name(gen),
                dset_train=newdata,
                dset_train_weight=self.pipeline.dset_train_weight,
            )

    def log(self, msg):
        print(f"ACT_LEARN: {msg}")
