# -*- coding: utf-8 -*-
"""
A demo of the kriging factory.
Kriging : cantilever beam model

References
* http://openturns.github.io/openturns/master/auto_meta_modeling/kriging_metamodel/plot_kriging_cantilever_beam.html#sphx-glr-auto-meta-modeling-kriging-metamodel-plot-kriging-cantilever-beam-py
* https://openturns.discourse.group/t/normalization-of-the-input-sample-in-krigingalgorithm-in-ot1-16/101

Output
------

+ Case #1: set bounds from the input sample
Lower and upper bounds of X_train:
Minimum: [3.82533e+08,6.04343,0.0081952,1.77756e-09]
Maximum: [3.82533e+10,604.343,0.81952,1.77756e-07]
Set bounds of scale optimization.
Optimized covariance=
SquaredExponential(scale=[1.93179e+10,305.193,0.413857,8.97666e-08], amplitude=[0.0801215])
+ Case #2: set bounds from quantiles of the input distribution
Set bounds of scale optimization.
Optimized covariance=
SquaredExponential(scale=[6.92678e+10,372.111,2.53469,1.59419e-07], amplitude=[0.255773])
+ Case #3: set bounds from the input sample
Set bounds of scale optimization.
Optimized covariance=
SquaredExponential(scale=[1.9127e+10,302.192,0.409818,8.88797e-08], amplitude=[0.0791811])
"""
import openturns as ot
import openturns.viewer as viewer

from openturns.usecases import cantilever_beam as cantilever_beam
import KrigingFactory as kg

ot.Log.Show(ot.Log.NONE)


def validate(kriging_result):
    optimized_covariance = kriging_result.getCovarianceModel()
    print("Optimized covariance=")
    print(optimized_covariance)
    krigingMetamodel = kriging_result.getMetaModel()

    # Validation
    sampleSize_test = 100
    X_test = cb.distribution.getSample(sampleSize_test)
    Y_test = cb.model(X_test)
    val = ot.MetaModelValidation(X_test, Y_test, krigingMetamodel)
    Q2 = val.computePredictivityFactor()[0]
    graph = val.drawValidation()
    graph.setTitle("Q2 = %.2f%%" % (100 * Q2))
    view = viewer.View(graph)
    return view


cb = cantilever_beam.CantileverBeam()

ot.RandomGenerator.SetSeed(0)

sampleSize_train = 10
X_train = cb.distribution.getSample(sampleSize_train)
Y_train = cb.model(X_train)

basis = ot.ConstantBasisFactory(cb.dim).build()
covarianceModel = ot.SquaredExponential(cb.dim)

basis = ot.ConstantBasisFactory(cb.dim).build()

# Case #1: set bounds from the input sample
print("+ Case #1: set bounds from the input sample")
x_range = X_train.getMax() - X_train.getMin()
scale_min_factor = 0.1  # Must be < 1, tune this to match your problem
scale_max_factor = 10.0  # Must be > 1, tune this to match your problem
minimum_scale_bounds = scale_min_factor * x_range
maximum_scale_bounds = scale_max_factor * x_range

print("Lower and upper bounds of X_train:")
print("Minimum:", minimum_scale_bounds)
print("Maximum:", maximum_scale_bounds)

factory = kg.KrigingFactory(X_train, Y_train, covarianceModel, basis)
algorithm = factory.buildFromBounds(minimum_scale_bounds, maximum_scale_bounds)
algorithm.run()
kriging_result = algorithm.getResult()
validate(kriging_result)

# Case #2: set bounds from quantiles of the input distribution
print("+ Case #2: set bounds from quantiles of the input distribution")

alphaLower = 0.01
factory = kg.KrigingFactory(X_train, Y_train, covarianceModel, basis)
algorithm = factory.buildFromQuantiles(cb.distribution, alphaLower)
algorithm.run()
kriging_result = algorithm.getResult()
validate(kriging_result)

# Case #3: set bounds from the input sample
print("+ Case #3: set bounds from the input sample")
scale_min_factor = 0.1
scale_max_factor = 10.0
factory = kg.KrigingFactory(X_train, Y_train, covarianceModel, basis)
algorithm = factory.buildFromSample(scale_min_factor, scale_max_factor)
algorithm.run()
kriging_result = algorithm.getResult()
validate(kriging_result)
