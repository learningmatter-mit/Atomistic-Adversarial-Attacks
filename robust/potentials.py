import torch as ch
import numpy as np
from math import sqrt


DEFAULT_EMIN = -np.inf
DEFAULT_EMAX = np.inf
DEFAULT_OFFSET = 0


class Potential:
    def __init__(self, *args, **kwargs):
        pass

    def get_energy(self, x):
        raise NotImplementedError

    def get_forces(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return x, self.get_energy(x), self.get_forces(x)


class NoisyPotential(Potential):
    def __init__(
        self,
        *args,
        energy_noise=0.0,
        forces_noise=0.0,
        **kwargs,
    ):
        self.enoise = energy_noise
        self.fnoise = forces_noise

        super(NoisyPotential, self).__init__(*args, **kwargs)

    def add_noise(self, values, std):
        return values + std * ch.randn_like(values)

    def __call__(self, x, add_noise=True):
        e = self.get_energy(x)
        f = self.get_forces(x)

        if add_noise:
            e = self.add_noise(e, self.enoise)
            f = self.add_noise(f, self.fnoise)

        return x, e, f


class LennardJones(NoisyPotential):
    def __init__(self, sigma, eps, **kwargs):
        self.sigma = sigma
        self.eps = eps
        super().__init__(**kwargs)

    def get_energy(self, x):
        r = sigma / x
        return 4 * self.eps * (ch.pow(r, 12) - ch.pow(r, 6))

    def get_forces(self, x):
        return 24 * self.eps * (2 * ch.pow(r, 13) - ch.pow(r, 7)) / self.sigma


class DoubleWell(NoisyPotential):
    def __init__(self, a, b, c=DEFAULT_OFFSET, **kwargs):
        self.a = a
        self.b = b
        self.c = c
        super(DoubleWell, self).__init__(**kwargs)

    def get_energy(self, x):
        return -self.a * (x ** 2) * (self.b - x) * (self.b + x) + self.c * x

    def get_forces(self, x):
        f = -2 * self.a * x * (2 * (x ** 2) - (self.b ** 2)) - self.c
        return f.reshape(-1, 1)

    @classmethod
    def from_zeros_and_depth(cls, zeros, depth, offset=DEFAULT_OFFSET, **kwargs):
        b = zeros * sqrt(2)
        a = depth * 4 / (b ** 4)
        c = offset / (2 * zeros)

        return cls(a, b, c=c, **kwargs)


class TwoDimensionalDoubleWell(NoisyPotential):
    def __init__(
        self,
        ax,
        bx,
        cx,
        ay,
        by,
        cy,
        **kwargs,
    ):
        self.px = DoubleWell(ax, bx, cx, **kwargs)
        self.py = DoubleWell(ay, by, cy, **kwargs)
        super(TwoDimensionalDoubleWell, self).__init__(**kwargs)

    def separate_input(self, xy):
        return xy[:, 0], xy[:, 1]

    def get_energy(self, xy):
        x, y = self.separate_input(xy)
        ex = self.px.get_energy(x)
        ey = self.py.get_energy(y)
        return ex + ey

    def get_forces(self, xy):
        x, y = self.separate_input(xy)
        fx = self.px.get_forces(x)
        fy = self.py.get_forces(y)
        return ch.cat([fx, fy], dim=1)


class MaskedPotential(NoisyPotential):
    def __init__(
        self,
        *args,
        emin=DEFAULT_EMIN,
        emax=DEFAULT_EMAX,
        **kwargs,
    ):
        self.emin = emin
        self.emax = emax
        super(MaskedPotential, self).__init__(*args, **kwargs)

    def __call__(self, x, clip=True, **kwargs):
        x, e, f = super().__call__(x, **kwargs)

        if not clip:
            return x, e, f

        # index of masked energies/forces
        i = (e < self.emax) & (e > self.emin)

        return x[i], e[i], f[i]


class MaskedDoubleWell(DoubleWell, MaskedPotential):
    def __init__(self, a, b, c=DEFAULT_OFFSET, emin=DEFAULT_EMIN, emax=DEFAULT_EMAX, *args, **kwargs):
        super(MaskedDoubleWell, self).__init__(
            *args, a=a, b=b, c=c, emin=emin, emax=emax, **kwargs
        )


class Masked2DDoubleWell(TwoDimensionalDoubleWell, MaskedPotential):
    def __init__(
        self,
        ax, bx, cx,
        ay, by, cy,
        *args,
        emin=DEFAULT_EMIN, emax=DEFAULT_EMAX,
        **kwargs,
    ):
        super(Masked2DDoubleWell, self).__init__(
            *args,
            ax=ax, bx=bx, cx=cx,
            ay=ay, by=by, cy=cy,
            emin=emin, emax=emax,
            **kwargs,
        )
