
import numpy as np
from hyperopt import hp, tpe, Trials, fmin
from SmoothFlow_Vreg import VectorOptimizer

def 

def opt_hyper(fullframes,
              n_range = 2, s_range = 5,
              losstype = 'MSE',
              max_evals = 100,
              st_reg = True):

    opt_vec = VectorOptimizer(fullframes, n_range, s_range, losstype)

    hyper_params = {}
    hyper_params['k1'] = hp.uniform('spatial', 0, 0.1)
    hyper_params['k2'] = hp.uniform('temporal', 0, 0.1)
    if st_reg:
        hyper_params['k3'] = hp.uniform('spatiotemporal', 0, 0.1)

    def _coefmake(k1 = 0, k2 = 0, k3 = 0):
        return np.array([1, k1, k2, k3])

    # objective function
    def _objective(args):
        coef = _coefmake(**args)
        opt_vec._vec_init()
        _, loss = opt_vec.optimize(num_iter = 5, coef = coef)
        return np.sum(loss)

    trials = Trials()
    best = fmin(
        _objective,
        hyper_params,
        algo = tpe.suggest,
        max_evals = max_evals,
        trials = trials,
        verbose = 1
    )
                
    return best, trials
        
