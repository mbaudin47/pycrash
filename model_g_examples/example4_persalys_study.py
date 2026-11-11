#!/usr/bin/env python

import openturns as ot
import persalys

exemple4_persalys_study = persalys.Study('exemple4_persalys_study')
dist_L = ot.Uniform(2.5, 2.6)
L = persalys.Input('L', 2.55, dist_L, 'Beam length (m)')
dist_F = ot.LogNormal(5.69881, 0.0997513, 0)
F = persalys.Input('F', 300, dist_F, 'Point load (N)')
dist_E = ot.Beta(0.9, 3.5, 6.5e+09, 7.5e+10)
E = persalys.Input('E', 70000000000, dist_E, 'Young\'s modulus (Pa)')
dist_I = ot.Beta(2.5, 4, 1.3e-07, 1.7e-07)
I = persalys.Input('I', 1.5e-07, dist_I, 'Moment of inertia (m4)')
Y = persalys.Output('Y', '')
inputs = [L, F, E, I]
outputs = [Y]
code = r'''
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
'''
ModèlePython_0 = persalys.PythonPhysicalModel('ModèlePython_0', inputs, outputs, code)
exemple4_persalys_study.add(ModèlePython_0)
tendanceCentrale_0 = persalys.MonteCarloAnalysis('tendanceCentrale_0', ModèlePython_0)
tendanceCentrale_0.setMaximumCoefficientOfVariation(0.01)
tendanceCentrale_0.setMaximumElapsedTime(60)
tendanceCentrale_0.setBlockSize(8)
tendanceCentrale_0.setSeed(0)
tendanceCentrale_0.setAllowFailedEvaluations(False)
exemple4_persalys_study.add(tendanceCentrale_0)
persalys.Study.Add(exemple4_persalys_study)
