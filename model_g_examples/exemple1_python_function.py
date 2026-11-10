import openturns as ot

# 1. Définir le modèle physique
def mon_modele(X):
    L, F, E, I = X
    f = (F * L**3) / (3 * E * I)
    return [f]

dimension = 4
input_function = ot.PythonFunction(dimension, 1, mon_modele)

# 2. Définir les lois d'entrée
distribution_L = ot.Normal(2.5, 0.05)
distribution_F = ot.Normal(3000.0, 300.0)
distribution_E = ot.Normal(2.1e11, 1e10)
distribution_I = ot.Normal(4.0e-6, 1e-7)

input_distribution = ot.JointDistribution([
    distribution_L, distribution_F, distribution_E, distribution_I
])

# 3. Propager les incertitudes
input_design = ot.MonteCarloExperiment(input_distribution, 1000)
input_sample = input_design.generate()
output_sample = input_function(input_sample)

if __name__ == "__main__":
    print("Échantillon de sortie (5 premières lignes) :")
    print(output_sample[:5])
