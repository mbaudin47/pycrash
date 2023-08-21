class SuperProjectionStrategy:
    def __init__(self, projectionStrategy):
        self.projectionStrategy = projectionStrategy

    def _repr_html_(self):
        """Get HTML representation."""
        html = ""
        html += "<ul>\n"
        html += f"  <li>coefficients: dimension= {self.projectionStrategy.getCoefficients().getDimension()}</li>\n"
        html += f"  <li>residual: {self.projectionStrategy.getResidual()}</li>\n"
        html += (
            f"  <li>relative error: {self.projectionStrategy.getRelativeError()}</li>\n"
        )
        html += f"  <li>measure: {self.projectionStrategy.getMeasure().getImplementation().getClassName()}</li>\n"
        html += f"  <li>experiment: {self.projectionStrategy.getExperiment().getClassName()}</li>\n"
        html += f"  <li>input sample: size= {self.projectionStrategy.getInputSample().getSize()} x dimension= {self.projectionStrategy.getInputSample().getDimension()}</li>\n"
        html += f"  <li>output sample: size= {self.projectionStrategy.getOutputSample().getSize()} x dimension= {self.projectionStrategy.getOutputSample().getDimension()}</li>\n"
        html += f"  <li>weights: dimension= {self.projectionStrategy.getWeights().getDimension()}</li>\n"
        html += "</ul>\n"

        return html
