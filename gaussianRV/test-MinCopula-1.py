import openturns as ot
from openturns.viewer import View

N=1000
x1=ot.Normal()
x2=ot.Normal()
x3=ot.Normal()
x4=ot.Normal()
x5=ot.Normal()
x6=ot.Normal()
x7=ot.Normal()
x8=ot.Normal()
x9=ot.Normal()
inputCollection=[x1,x2,x3,x4,x5,x6,x7,x8,x9]
mycopula12 = ot.MinCopula(2)
mycopula35 = ot.MinCopula(3)
mycopula69 = ot.IndependentCopula(4)
mycopula = ot.ComposedCopula([mycopula12,mycopula35,mycopula69])
inputDistribution = ot.ComposedDistribution(inputCollection,mycopula)
X=inputDistribution.getSample(N)
myGraph = ot.Graph('Pairs', ' ', ' ', True, '')
myPairs = ot.Pairs(X, 'Pairs example', X.getDescription(), 'blue', 'bullet')
myGraph.add(myPairs)
View(myGraph).show()

from numpy import array
from numpy.linalg import eig
from numpy.linalg import cholesky

rho=0.9999999999999999
R=array([[1,rho,0.5],[rho,1,0.5],[0.5,0.5,1]])
w,v = eig(R)
print w
print cholesky(R)
