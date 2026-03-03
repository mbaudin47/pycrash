"""
References
----------
- Lüthen, N., Marelli, S., & Sudret, B. (2021). 
  Sparse polynomial chaos expansions: Literature survey and benchmark.
  SIAM/ASA Journal on Uncertainty Quantification, 9 (2), 593-649.
"""
# %%
import openturns as ot

# %%
## Algorithme PCE plein
# Input : input_sample, wX, output_sample, distribution, maximumBasisSize
# Output : ot.FunctionalChaosResult
# 
transformation = ot.DistributionTransformation(
	self.distribution, self.basis.getMeasure()
)
standard_input = transformation(self.input_sample)
if self.selectedIndices is None:
	indices = ot.Indices(self.maximumBasisSize)
	indices.fill()
	functions = [self.basis.build(i) for i in indices]
else:
	indices = self.selectedIndices
	functions = [self.basis.build(i) for i in indices]
designProxy = ot.DesignProxy(standard_input, functions)
if self.wX is None:
	leastSquaresMethod = ot.LeastSquaresMethod.Build(
		self.leastSquaresMethod, designProxy, range(len(indices))
	)
else:
	leastSquaresMethod = ot.LeastSquaresMethod.Build(
		self.leastSquaresMethod, designProxy, self.wX, range(len(indices))
	)
outputDimension = self.output_sample.getDimension()
coefficients = ot.Sample(len(indices), outputDimension)
for j in range(outputDimension):
	coeffsJ = leastSquaresMethod.solve(
		self.output_sample.getMarginal(j).asPoint()
	)
	for i in range(len(indices)):  # coefficients[:,j] = coeffsJ
		coefficients[i, j] = coeffsJ[i]
# Create the result
# The physical model is unknown in this case ...
physicalModel = ot.Function()
# ... which implies that the composed model is unknown in this case
composedModel = ot.Function()
residualsPoint = [-1.0]
relativeErrorsPoint = [-1.0]
self.result = ot.FunctionalChaosResult(
	self.input_sample,
	self.output_sample,
	self.distribution,
	transformation,
	transformation.inverse(),
	self.basis,
	indices,
	coefficients,
	functions,
	residualsPoint,
	relativeErrorsPoint,
)

# %%
## Algorithme Orthogonal Matching Pursuit (OMP)
# Input : input_sample, wX, output_sample, distribution, maximum_basis_dimension
# Output : ot.FunctionalChaosResult
transformation = ot.DistributionTransformation(
	self.distribution, self.basis.getMeasure()
)
standard_input = transformation(self.input_sample)

# Initialisation
list_of_active_functions = [0]
output_mean = self.output_sample.getMean()
coefficients = output_mean.asPoint()
residuals = ot.Sample(output_sample)
sample_size = standard_input.getSize()
maximum_basis_dimension = 10  # Set by the user
for i in range(maximum_basis_dimension):
    # Find candidate with maximum absolute correlation with the residual
    maximum_absolute_residual = 0.0
    best_basis_function_index = None
    for j in range(maximum_basis_dimension):
        current_basis_function = self.basis.build(j)
        basis_function_value = current_basis_function(standard_input)
        # What if the residuals has (output) dimension > 1 ?
        current_absolute_correlation = abs(residuals.dot(basis_function_value))
        if current_absolute_correlation > maximum_absolute_residual:
            best_basis_function_index = j
            maximum_absolute_residual = current_absolute_correlation
    # Add the best candidate to the active set
    list_of_active_functions.append(best_basis_function_index)
    # Update the coefficients
	functions = [self.basis.build(i) for i in list_of_active_functions]
	designProxy = ot.DesignProxy(standard_input, functions)
	leastSquaresMethod = ot.LeastSquaresMethod.Build(
		self.leastSquaresMethod, designProxy, 
	)
	outputDimension = self.output_sample.getDimension()
	coefficients = ot.Sample(len(indices), outputDimension)
	for j in range(outputDimension):
		coeffsJ = leastSquaresMethod.solve(
			self.output_sample.getMarginal(j).asPoint()
		)
		for i in range(len(indices)):
			coefficients[i, j] = coeffsJ[i]
	

# Réserve
# Create the result
# The physical model is unknown in this case ...
physicalModel = ot.Function()
# ... which implies that the composed model is unknown in this case
composedModel = ot.Function()
residualsPoint = [-1.0]
relativeErrorsPoint = [-1.0]
self.result = ot.FunctionalChaosResult(
	self.input_sample,
	self.output_sample,
	self.distribution,
	transformation,
	transformation.inverse(),
	self.basis,
	indices,
	coefficients,
	functions,
	residualsPoint,
	relativeErrorsPoint,
)
