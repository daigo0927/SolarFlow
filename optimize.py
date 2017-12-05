
import numpy as np
from hyperopt import hp, tpe, Trials, fmin
from SmoothFlow_Vreg import smoothflow_withVreg
from SmoothInterp import fullinterp

def opt_hyper(data,
              n_range = 2, s_range = 7,
              losstype = 'MSE',
              max_evals = 100,
              st_reg = True):

    frames_train_for = data['shade'][::2]
    frames_train_back = data['shade'][::-2]

    hyper_params = {}
    hyper_params['k1'] = hp.uniform('spatial', 0, 0.1)
    hyper_params['k2'] = hp.uniform('temporal', 0, 0.1)
    if st_reg:
        hyper_params['k3'] = hp.uniform('spatiotemporal', 0, 0.1)

    # objective function
    def solar_reconst(args):
        coef = _makecoef(**args)
        vec_for = smoothflow_withVreg(fullframes = frames_train_for,
                                      coef = coef,
                                      neighbor_range = n_range,
                                      search_range = s_range,
                                      losstype = losstype)
        vec_back = smoothflow_withVreg(fullframes = frames_train_back,
                                       coef = coef,
                                       neighbor_range = n_range,
                                       search_range = s_range,
                                       losstype = losstype)
        shade_reconst = fullinterp(frames_train_for, vec_for, vec_back, 2)
        total_reconst = aaa # サイズ合わせ
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

def _makecoef(k1 = 0, k2 = 0, k3 = 0):
    return np.array([1, k1, k2, k3])


