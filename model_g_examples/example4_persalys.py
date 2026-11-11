# Physical model implementation for Persalys (Python interface)

def _exec(L, F, E, I):
    """Evaluation function called by Persalys.

    Parameters
    ----------
    L : float
        Beam length (m).
    F : float
        Point load (N).
    E : float
        Young's modulus (Pa).
    I : float
        Moment of inertia (m4).

    Returns
    -------
    Y : float
        Beam tip deflection (m).
    """
    Y = (F * L**3) / (3.0 * E * I)
    return Y