# Implémentation du modèle physique pour Persalys (Interface Python)

def _exec(L, F, E, I):
    """
    Evaluation function called by Persalys.
    
    Inputs:
        L : Beam length (m)
        F : Point load (N)
        E : Young's modulus (Pa)
        I : Moment of inertia (m4)
        
    Output:
        Y : Beam tip deflection (m)
    """
    Y = (F * L**3) / (3.0 * E * I)
    return Y
