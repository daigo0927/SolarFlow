
import numpy as np
from hyperopt import hp, tpe, Trials, fmin
from SmoothFlow_Vreg import smoothflow_withVreg

def opt(fullframes, spatiotemporal = True):
    

    def _objective(k1 = 0, k2 = 0, k3 = 0):
    

    
        
