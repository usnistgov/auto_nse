import numpy as np
from typing import List, Union, Tuple
from bumps.parameter import Parameter
from pydantic.v1.dataclasses import dataclass, Field
from dataclasses import field
from numpy_encoder import NumpyNDArray

@dataclass
class AutoPar:
    name: str
    value: float
    bounds: Union[Tuple[float, float], None] = None

@dataclass
class AutoFunc:
    """Base function class for autonomous experiments. Designed to be a serializable
        version of the Bumps Curve definition (without including data)"""
    x: NumpyNDArray

    @staticmethod
    def func(x):

        return np.zeros_like(x)

    def __post_init_post_parse__(self):

        pardict = {}
        for attr in [a for a in dir(self) if not a.startswith('__')]:
            obj = getattr(self, attr)
            if isinstance(obj, AutoPar):
                newpar = Parameter(name=obj.name, value=obj.value)
                if obj.bounds is not None:
                    newpar.range(*obj.bounds)
                pardict[attr] = newpar

        self.parameters = pardict

@dataclass
class Gaussian(AutoFunc):
    """Gaussian function"""
    A: AutoPar
    x0: AutoPar
    sigma: AutoPar
    name: str = 'gaussian'

    @staticmethod
    def func(x, A, x0, sigma):

        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

@dataclass
class GaussianPlusBackground(AutoFunc):
    """Gaussian function with background"""
    A: AutoPar
    x0: AutoPar
    sigma: AutoPar
    bkgd: AutoPar
    name: str = 'gaussian_background'

    @staticmethod
    def func(x, A, x0, sigma, bkgd):

        return super().func(x, A, x0, sigma) + bkgd

@dataclass
class NSE_echo(AutoFunc):
    A: AutoPar
    I0: AutoPar
    phi0: AutoPar
    sigma: AutoPar
    T: AutoPar
    name: str = 'nse_echo'

    @staticmethod
    def func(x, I0, A, phi0, sigma, T):
        """NSE echo function
        Inputs:
        I0 -- average count rate [n/s]
        A -- amplitude of echo [n/s]
        phi0 -- phase of maximum amplitude [deg]
        sigma -- width of echo envelope [deg]
        T -- period of phase oscillation [deg]
        """
        
        phi = x
        return I0 - A * np.exp(-(phi - phi0) ** 2 / (2 * sigma ** 2)) * np.cos(np.radians(360. / T * (phi - phi0)))

function_types = Union[Gaussian, GaussianPlusBackground, NSE_echo]

if __name__ == '__main__':

    from bumps.names import Curve
    import json
    from numpy_encoder import numpy_encoder

    x = np.linspace(-30, 30, 101)
    Apar = AutoPar('A', 10, [0, 20])
    x0par = AutoPar('x0', 10, [0, 20])
    sigmapar = AutoPar('sigma', 2, None)
    bkgd = AutoPar('bkgd', 10, [0, 20])

    g = GaussianPlusBackground(x, Apar, x0par, sigmapar, bkgd)

    xdata = [0, 1, 2]
    ydata = [6, 7, 8]
    dydata = [3, 4, 5]

    m = Curve(g.func, xdata, ydata, dydata, name=g.name, **g.parameters)

    #print(m.parameters())

    s = json.dumps(g, indent=4, default=numpy_encoder)
#    print(s)

    g = GaussianPlusBackground(**json.loads(s))
 #   print(g)








