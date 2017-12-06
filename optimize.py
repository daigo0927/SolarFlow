
import numpy as np
from hyperopt import hp, tpe, Trials, fmin
from SmoothFlow_Vreg import smoothflow_withVreg
from SmoothInterp import fullinterp

def opt_hyper(data,
              limit_frame = 999,
              n_range = 2, s_range = 7,
              losstype = 'MSE',
              max_evals = 100,
              st_reg = True):

    if limit_frame%2 == 0:
        raise ValueError('given data must have odd length')
    for key in ['shade', 'crop', 'outer']:
        data[key] = np.array(data[key])[:limit_frame]
        
    num_frames, frame_size_origin, _ = data['shade'].shape
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
        
        _, frame_size, _ = shade_reconst.shape
        rem = int((frame_size_origin - frame_size)/2)
        # crop and adjust for the interpolated result
        outer_ = data['outer'][:, rem:frame_size+rem, rem:frame_size+rem]
        total_reconst = (1 - shade_reconst)*outer_
        total_ = data['crop'][:, rem:frame_size+rem, rem:frame_size+rem]
        
        return np.mean((total_reconst - total_)**2)

    trials = Trials()
    best = fmin(
        solar_reconst,
        hyper_params,
        algo = tpe.suggest,
        max_evals = max_evals,
        trials = trials,
        verbose = 1
    )
                
    return best, trials

def _makecoef(k1 = 0, k2 = 0, k3 = 0):
    return np.array([1, k1, k2, k3])


