#    The ImpedanceFitter is a package to fit impedance spectra to
#    equivalent-circuit models using open-source software.
#
#    Copyright (C) 2018, 2019 Leonard Thiele, leonard.thiele[AT]uni-rostock.de
#    Copyright (C) 2018, 2019, 2020 Julius Zimmermann, julius.zimmermann[AT]uni-rostock.de
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Adapted from https://impedancefitter.readthedocs.io/en/v2.0.0/examples/statistics.html

import numpy as np
from collections import OrderedDict
from impedancefitter import get_equivalent_circuit_model
from matplotlib import rcParams
import openturns as ot

rcParams['figure.figsize'] = [20, 10]

# parameters
f = np.logspace(1, 8)
omega = 2. * np.pi * f
R = 1000.0
C = 1e-6

data = OrderedDict()
data['f'] = f

# Part 1 : without OpenTURNS
sample_size = 10
np.random.seed(1)
model = "parallel(R, C)"
m = get_equivalent_circuit_model(model)
# generate random samples
for i in range(sample_size):
    Ri = 0.05 * R * np.random.randn() + R
    Ci = 0.05 * C * np.random.randn() + C

    Z = m.eval(omega=omega, R=Ri, C=Ci)
    # add some noise
    Z += np.random.randn(Z.size)

    data['real' + str(i)] = Z.real
    data['imag' + str(i)] = Z.imag

print(data)

# Part 2 : With OpenTURNS
sample_size = 10
ot.RandomGenerator.SetSeed(1)
model = "parallel(R, C)"
m = get_equivalent_circuit_model(model)

# Distribution : use a Normal with coefficient of variation equal to 0.05
R = ot.Normal(1000.0, 0.05 * 1000.0)
C = ot.Normal(1e-6, 0.05 * 1e-6)
Noise = ot.Normal(0.0, 1.0)
distribution = ot.ComposedDistribution([R, C, Noise])

# Define model [R, C, output_noise, omega] -> [Zreal, Zimag]
def model(X):
    R, C, output_noise, omega = X
    Z = m.eval(omega=omega, R=R, C=C)
    Zreal = Z.real + output_noise
    Zimag = Z.imag + output_noise
    Z = [Zreal, Zimag]
    return Z
dimension_input = 4
dimension_output = 2
model_Py = ot.PythonFunction(dimension_input, dimension_output, model)
X = [1000.0, 1.e-6, 0.1, 500.0]
Z = model_Py(X)
print("X=", X)
print("Z=", Z)

# Set omega as parameter
indices = [3]
values = [500.0] # Constant omega value
parametric_model = ot.ParametricFunction(model_Py, indices, values)
X = [1000.0, 1.e-6, 0.1]
Z = parametric_model(X)
print("X=", X)
print("Z=", Z)

# Generate one sample
input_random_vector = ot.RandomVector(distribution)
random_vector_Z = ot.CompositeRandomVector(parametric_model, input_random_vector)
sampleZ = random_vector_Z.getSample(sample_size)
print(sampleZ)

# Generate a collection of samples corresponding to all values 
# of omega
sample_list = []
for omega_value in omega:
    parametric_model.setParameter([omega_value])
    input_random_vector = ot.RandomVector(distribution)
    random_vector_Z = ot.CompositeRandomVector(parametric_model, input_random_vector)
    sampleZ = random_vector_Z.getSample(sample_size)
    sample_list.append(sampleZ)
