#!/usr/bin/env python3

"""
Simulation du cas-test zzzz159a
===============================

Solution de référence obtenue avec (200000, 200, 2000) en appliquant
une déformation homogène de 0. à 0.005 sur 20 pas de temps.

On obtient les "courbes expérimentales" sigma(t) et eps_p(t).

À partir des ces "courbes expérimentales", on cherche à retrouver les trois
paramètres E, sigma_y, E_T avec :

- 50.e3 < E < 500.e3, valeur initiale 100.e3.
- 500 < E_T < 10000, valeur initiale 1000.
- 5 < sigma_y < 500, valeur initiale 30.

Documentations :

- https://www.code-aster.org/V2/doc/default/fr/man_v/v1/v1.01.159.pdf

- https://www.code-aster.org/V2/doc/default/fr/man_r/r5/r5.03.02.pdf
"""


import numpy as np


class VmisIsotLine:
    """Simulator for VMIS_ISOT_LINE"""

    def __init__(self, E, sigma_y, E_T):
        self.E = E
        self.sigma_y = sigma_y
        self.E_T = E_T

    def integr(self, epsi):
        """Returns sigma, eps_p."""
        if epsi < self.sigma_y / self.E:
            sigma = self.E * epsi
            intvar = 0.0
        else:
            sigma = self.sigma_y + self.E_T * (epsi - self.sigma_y / self.E)
            intvar = epsi - sigma / self.E
        return sigma, intvar


reference = VmisIsotLine(200000.0, 200.0, 2000.0)

eps_max = 5e-3
incr = eps_max / 20.0
hist = np.arange(0.0, eps_max + incr, incr)

for load in hist:
    sigma, v1 = reference.integr(load)
    print(f"{load:13.6f} {sigma:13.6f} {v1:13.6e}")
