import openturns as ot

# 1. Définition de la classe dérivée pour la fonction g-Sobol paramétrée
class GSobolFunction(ot.OpenTURNSPythonFunction):
    def __init__(self, a=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]):
        self.a_ = ot.Point(a)
        dimension = len(a)
        super().__init__(dimension, 1)  # Appel du constructeur de la classe mère
        self.setInputDescription([f"X{i}" for i in range(dimension)])
        self.setOutputDescription(["Y"])

    def _exec(self, X):
        val = 1.0
        for i in range(len(self.a_)):
            val *= (abs(4.0 * X[i] - 2.0) + self.a_[i]) / (1.0 + self.a_[i])
        return [val]

if __name__ == "__main__":
    # 2. Instanciation avec un vecteur de poids personnalisé et encapsulation
    g_sobol_obj = GSobolFunction(a=[0.0, 1.0, 2.0, 5.0])
    modele_gsobol = ot.Function(g_sobol_obj)

    # 3. Définition du problème probabiliste (Uniforme(0, 1)^d)
    dimension = modele_gsobol.getInputDimension()
    distribution = ot.JointDistribution([ot.Uniform(0.0, 1.0)] * dimension)

    # 4. Évaluation sur un plan d'expériences
    sample_input = distribution.getSample(100)
    sample_output = modele_gsobol(sample_input)

    print("Échantillon de sortie g-Sobol (5 premières lignes) :")
    print(sample_output[:5])
