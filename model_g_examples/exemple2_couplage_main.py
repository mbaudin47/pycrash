import sys
import openturns as ot
import openturns.coupling_tools as ct

# 1. Définition de la classe dérivée paramétrée pour le couplage
class PoutreEncastreeCouplage(ot.OpenTURNSPythonFunction):
    def __init__(self, infile_template="input_template.txt", infile="input.txt", outfile="output.txt", program="external_program.py"):
        super().__init__(4, 1)  # Appel du constructeur de la classe mère
        self.setInputDescription(['L', 'F', 'E', 'I'])
        self.setOutputDescription(['f'])

        # Attributs de configuration encapsulés dans le constructeur
        self.infile_template_ = infile_template
        self.infile_ = infile
        self.outfile_ = outfile
        self.program_ = program
        self.tokens_in_ = ["@L@", "@F@", "@E@", "@I@"]
        self.tokens_out_ = ["f="]

    def _exec(self, X):
        # a. Remplacement des balises dans le fichier d'entrée patron
        ct.replace(self.infile_template_, self.infile_, self.tokens_in_, X)

        # b. Exécution du binaire/code de calcul externe paramétré
        cmd = f"{sys.executable} {self.program_} {self.infile_}"
        ct.execute(cmd)

        # c. Extraction du résultat dans le fichier de sortie
        Y = ct.get(self.outfile_, tokens=self.tokens_out_)
        return Y

if __name__ == "__main__":
    # 2. Instanciation du modèle en personnalisant le programme et les fichiers, puis encapsulation dans ot.Function
    modele_physique = PoutreEncastreeCouplage(
        infile_template="input_template.txt",
        infile="input.txt",
        outfile="output.txt",
        program="external_program.py"
    )
    input_function = ot.Function(modele_physique)

    # 3. Définition du problème probabiliste et propagation d'incertitudes
    distribution_L = ot.Normal(2.5, 0.05)
    distribution_F = ot.Normal(3000.0, 300.0)
    distribution_E = ot.Normal(2.1e11, 1e10)
    distribution_I = ot.Normal(4.0e-6, 1e-7)

    input_distribution = ot.JointDistribution([
        distribution_L, distribution_F, distribution_E, distribution_I
    ])

    input_design = ot.MonteCarloExperiment(input_distribution, 10)
    input_sample = input_design.generate()
    output_sample = input_function(input_sample)

    print("Échantillon de sortie (5 premières lignes) :")
    print(output_sample[:5])
