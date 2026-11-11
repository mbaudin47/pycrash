import openturns as ot


class GSobolFunction(ot.OpenTURNSPythonFunction):
    """
    Parameterized g-Sobol function implementation for OpenTURNS.

    Parameters
    ----------
    a : list of float, default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]
        Vector of parameters weighting the importance of each input dimension.
    """

    def __init__(self, a=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]):
        self.a_ = ot.Point(a)
        dimension = len(a)
        super().__init__(dimension, 1)  # Call parent class constructor
        self.setInputDescription([f"X{i}" for i in range(dimension)])
        self.setOutputDescription(["Y"])

    def _exec(self, X):
        """
        Evaluate the g-Sobol function at a given input point.

        Parameters
        ----------
        X : sequence of float
            Input evaluation vector.

        Returns
        -------
        list of float
            Output value of the g-Sobol function evaluated at X.
        """
        val = 1.0
        for i in range(len(self.a_)):
            val *= (abs(4.0 * X[i] - 2.0) + self.a_[i]) / (1.0 + self.a_[i])
        return [val]


if __name__ == "__main__":
    # 2. Instantiation with a custom weight vector and encapsulation
    g_sobol_obj = GSobolFunction(a=[0.0, 1.0, 2.0, 5.0])
    gsobol_model = ot.Function(g_sobol_obj)

    # 3. Definition of the probabilistic problem (Uniform(0, 1)^d)
    dimension = gsobol_model.getInputDimension()
    distribution = ot.JointDistribution([ot.Uniform(0.0, 1.0)] * dimension)

    # 4. Evaluation on a design of experiments
    sample_input = distribution.getSample(100)
    sample_output = gsobol_model(sample_input)

    print("g-Sobol output sample (first 5 rows):")
    print(sample_output[:5])