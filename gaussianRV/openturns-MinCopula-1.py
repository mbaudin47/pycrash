import openturns as ot
from matplotlib import pyplot as plt
from openturns.viewer import View
copula = ot.MinCopula(2)
copula.setDescription(['$y_1$', '$y_2$'])
pdf_graph = copula.drawPDF()
cdf_graph = copula.drawCDF()
fig = plt.figure(figsize=(10, 4))
plt.suptitle(str(copula))
pdf_axis = fig.add_subplot(121)
cdf_axis = fig.add_subplot(122)
View(pdf_graph, figure=fig, axes=[pdf_axis], add_legend=False)
View(cdf_graph, figure=fig, axes=[cdf_axis], add_legend=False)
pdf_axis.set_aspect('equal')
cdf_axis.set_aspect('equal')

#
N=10000
from pylab import plot, xlabel, ylabel, show
x1=ot.Normal()
x2=ot.Normal()
mycopula = ot.MinCopula(2)
inputCollection=[x1,x2]
inputDistribution = ot.ComposedDistribution(inputCollection,mycopula)
X=inputDistribution.getSample(N)
plot(X[:,0],X[:,1])
xlabel("X1")
ylabel("X2")
show()
#
#
from pylab import plot, xlabel, ylabel, show
rho=0.9999
x1=ot.Normal()
x2=ot.Normal()
R=ot.CorrelationMatrix([[1.,rho],[rho,1.]])
mycopula = ot.NormalCopula(R)
inputCollection=[x1,x2]
inputDistribution = ot.ComposedDistribution(inputCollection,mycopula)
X=inputDistribution.getSample(N)
plot(X[:,0],X[:,1],"b.")
xlabel("X1")
ylabel("X2")
show()
#
hist = ot.VisualTest.DrawHistogram(X[:,0])
View(hist).show()
#
hist = ot.VisualTest.DrawHistogram(X[:,1])
View(hist).show()

#
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



R = ot.CorrelationMatrix(3)
R[0, 1] = 0.5
R[0, 2] = 0.25
collection = [ot.FrankCopula(3.0), ot.NormalCopula(R), ot.ClaytonCopula(2.0)]
copula = ot.ComposedCopula(collection)
