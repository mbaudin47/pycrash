#!/usr/bin/env python
# coding: utf-8

# L'objectif de ce NB est de vérifier la distribution de la solution du problème de moindres carrés linéaires. Plus précisément, on vérifie que la covariance de la loi gaussienne associée à la méthode `getParameterPosterior` est correcte pour la classe `GaussianLinearCalibration`.
# 
# On considère ici un modèle *exactement* linéaire. On essaye de calculer l'estimateur MAP par une décomposition de Cholesky puis de calculer la matrice de covariance.

# ## Generate the observations

# In[1]:


import openturns as ot


# In[2]:


ot.RandomGenerator.SetSeed(0)
ot.Log.Show(ot.Log.ALL)


# In[3]:
def modelLineaire(X):
    x,theta1,theta2,theta3 = X
    y = theta1 + theta2*x + theta3*x**2
    return [y]


# In[4]:
g = ot.PythonFunction(4, 1, modelLineaire) 


# In[5]:
descr = ["Theta1", "Theta2", "Theta3"]

# In[6]:
trueParameter = ot.Point([11.11,6.6,-9.9])


# In[7]:
parameterDimension = trueParameter.getDimension()

# Create the joint input distribution function.

# In[8]:
X = ot.Uniform()
Theta1 = ot.Dirac(trueParameter[0])
Theta2 = ot.Dirac(trueParameter[1])
Theta3 = ot.Dirac(trueParameter[2])

X.setDescription(["X"])
Theta1.setDescription(["Theta1"])
Theta2.setDescription(["Theta2"])
Theta3.setDescription(["Theta3"])

inputRandomVector = ot.ComposedDistribution([X, Theta1, Theta2, Theta3])

# In[9]:
calibratedIndices = [1,2,3]
model = ot.ParametricFunction(g, calibratedIndices, trueParameter)

# Generate observation noise.

# In[11]:
outputObservationNoiseSigma = 2. # (Pa)
observationOutputNoise = ot.Normal(0.,outputObservationNoiseSigma)


# ## Gaussian linear calibration

# Define the covariance matrix of the output Y of the model.

# In[12]:
errorCovariance = ot.CovarianceMatrix(1)
errorCovariance[0,0] = outputObservationNoiseSigma**2

# Defined the covariance matrix of the parameters $\theta$ to calibrate.

# In[13]:
sigmaTheta1 = 0.1 * trueParameter[0]
sigmaTheta2 = 0.1 * trueParameter[1]
sigmaTheta3 = 0.1 * trueParameter[2]

# In[14]:
parameterCovariance = ot.CovarianceMatrix(3)
parameterCovariance[0,0] = sigmaTheta1**2
parameterCovariance[1,1] = sigmaTheta2**2
parameterCovariance[2,2] = sigmaTheta3**2
parameterCovariance

# In[15]:
size = 10

# In[16]:
# 1. Generate exact outputs
inputSample = inputRandomVector.getSample(size)
outputStress = g(inputSample)
# 2. Add noise
sampleNoiseH = observationOutputNoise.getSample(size)
outputObservations = outputStress + sampleNoiseH
# 3. Calibrate
inputObservations = inputSample[:,0]

# In[17]:
candidate = ot.Point([12.,7.,-8])

# In[18]:
algo = ot.GaussianLinearCalibration(model, inputObservations, outputObservations,                                     candidate, parameterCovariance, errorCovariance)
algo.run()
calibrationResult = algo.getResult()

# ## Analysis of the results
# The `getParameterMAP` method returns the maximum of the posterior distribution of $\theta$.
# In[19]:
thetaStar = calibrationResult.getParameterMAP()
print("thetaStar")
print(thetaStar)

# In[20]:
thetaPosterior = calibrationResult.getParameterPosterior()

# In[21]:
covarianceThetaStar = thetaPosterior.getCovariance()
print("covarianceThetaStar")
print(covarianceThetaStar)

# ## Calibration based on Kalman matrix
# In[22]:
parameterDimension = candidate.getDimension()

# In[23]:
model.setParameter(candidate)

# In[24]:
modelObservations = model(inputObservations)

# In[25]:
transposedGradientObservations = ot.Matrix(parameterDimension,size)
for i in range(size):
    g = model.parameterGradient(inputObservations[i])
    transposedGradientObservations[:,i] = g

# In[26]:
gradientObservations = transposedGradientObservations.transpose()
print("gradientObservations")
print(gradientObservations)

# In[27]:
deltay = outputObservations - modelObservations
deltay = deltay.asPoint()

# Compute inverses of B and R.
# In[28]:
B = ot.CovarianceMatrix(parameterCovariance)
IB = ot.IdentityMatrix(parameterDimension)
invB = B.solveLinearSystem(IB)
print("invB")
print(invB)

# In[29]:
R = ot.CovarianceMatrix(size)
for i in range(size):
    R[i,i] = errorCovariance[0,0]
print("R")
print(R)
IR = ot.IdentityMatrix(size)
invR = R.solveLinearSystem(IR)
print("invR")
print(invR)

# Calcule $A^{-1} = B^{-1} + J^T R^{-1} J = B^{-1} + J^T (J^T R^{-1})^T$.
# Soit $C =J^T R^{-1}$. 
# Cela implique $A^{-1} = B^{-1} + J^T C^T$.

# In[30]:
C = gradientObservations.transpose() * invR
print("C")
print(C)

# In[31]:
invA = invB + C * gradientObservations
print("invA")
print(invA)

# Calcule $K = A J^T R^{-1} = A C = (A^{-1})^{-1} C$.
# In[32]:
K = invA.solveLinearSystem(C)
print("K")
print(K)

# Calcule $\hat{\theta} = \mu  + K (y - H(\mu ))$.
# In[33]:
thetaStar = candidate + K * deltay
print("thetaStar")
print(thetaStar)

# ## Autre calcul de la matrice de covariance
# Un autre moyen de calculer la matrice de covariance.

# In[36]:
L = IB - K * gradientObservations
print("L")
print(L)

# In[37]:
covarianceThetaStar = K * R * K.transpose() + L * B * L.transpose()
print("covarianceThetaStar")
print(covarianceThetaStar)
