import openturns as ot
from matplotlib import pyplot as plt
from openturns.viewer import View
rho=-0.9999999999
R=ot.CorrelationMatrix([[1.,rho],[rho,1.]])
copula = ot.NormalCopula(R)
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