
import numpy as np
from hyperopt import hp, tpe, Trials, fmin
from SmoothFlow_Vreg import smoothflow_withVreg
from SmoothInterp import fullinterp
from misc.utils import LinearInterp

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
        # crop and adjust for the shape of interpolated result
        outer_ = data['outer'][:, rem:frame_size+rem, rem:frame_size+rem]
        total_reconst = (1 - shade_reconst)*outer_
        total_ = data['crop'][:, rem:frame_size+rem, rem:frame_size+rem]
        
        return np.mean(np.sqrt((total_reconst - total_)**2))

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

# interpolation class, integrate methods
# validation:True is the one-drop validation
class Interpolater(object):

    def __init__(self, data, limit_frame = 999, validation = None, fineness = 15):

        limit_frame = min(len(data['crop']), limit_frame)
        
        self.data = {}
        self.data['crop'] = data['crop'][:limit_frame]
        self.data['outer'] = np.array(data['outer'])[:limit_frame]
        self.data['outer_fine'] = np.array(data['outer_fine'])[:(limit_frame-1)*15+1]
        self.data['shade'] = data['shade'][:limit_frame]

        self.validation = validation
        if self.validation is not None:
            if (limit_frame-1)%self.validation != 0:
                raise ValueError('limit_frame must be adjust to validation number')
            self.n_range = 2
            self.s_range = 5+2*(self.validation-1)
            self.fineness = self.validation
            self.shade = self.data['shade'][:limit_frame:self.validation]
            self.outer_fine = self.data['outer']
        else:
            self.n_range = 2
            self.s_range = 5
            assert 15%fineness == 0, 'fineness should be a divisor number of 15'
            self.fineness = fineness
            self.shade = self.data['shade']
            self.outer_fine = self.data['outer_fine'][::int(15/fineness)]

    def linear_interp(self):
        lin = LinearInterp(data = self.shade)
        lin.interp(fineness = self.fineness)
        shade_fine = lin.result
        total_fine = (1 - shade_fine)*self.outer_fine
        return total_fine

    def flow_interp(self, losstype):
        forflow, backflow = doubleflow_withVreg(fullframes = self.shade,
                                                coef = np.array([1., 0., 0., 0.]),
                                                neighbor_range = self.n_range,
                                                search_range = self.s_range,
                                                losstype = losstype)
        shade_fine = fullinterp(self.shade, forflow, backflow, self.fineness)
        total_fine = (1 - shade_fine)*croparray(self.outer_fine, shade_fine)
        return total_fine
        
    def flow_interp_doubleregs(self, losstype, max_evals):
        # get the best hyper parameter
        best_hyperparams, _ = opt_hyper(self.data, losstype = losstype,
                                        max_evals = max_evals, st_reg = False)
        k1 = best_hyperparams['spatial']
        k2 = best_hyperparams['temporal']
        forflow, backflow = doubleflow_withVreg(fullframes = self.shade,
                                                coef = np.array([1., k1, k2, 0.]),
                                                neighbor_range = self.n_range,
                                                search_range = self.s_range,
                                                losstype = losstype)
        shade_fine = fullinterp(self.shade, forflow, backflow, self.fineness)
        total_fine = (1 - shade_fine)*croparray(self.outer_fine, shade_fine)
        return total_fine, np.array([1., k1, k2, 0.])

    def flow_interp_tripleregs(self, losstype, max_evals):
        # get the best hyper parameter
        best_hyperparams, _ = opt_hyper(self.data, losstype = losstype,
                                        max_evals = max_evals, st_reg = True)
        k1 = best_hyperparams['spatial']
        k2 = best_hyperparams['temporal']
        k3 = best_hyperparams['spatiotemporal']
        forflow, backflow = doubleflow_withVreg(fullframes = self.shade,
                                                coef = np.array([1., k1, k2, k3]),
                                                neighbor_range = self.n_range,
                                                search_range = self.s_range,
                                                losstype = losstype)
        shade_fine = fullinterp(self.shade, forflow, backflow, self.fineness)
        total_fine = (1 - shade_fine)*croparray(self.outer_fine, shade_fine)
        return total_fine, np.array([1., k1, k2, k3])

# crop and make target_attay have same shape of shape_array
def croparray(target_array, shape_array):

    num_frames, frame_size_origin, _ = target_array.shape
    num_frames_, frame_size, _ = shape_array.shape
    assert num_frames == num_frames, 'given array have invalid num_frames'

    rem = int((frame_size_origin - frame_size)/2)
    return target_array[:, rem:frame_size+rem, rem:frame_size+rem]

def doubleflow_withVreg(fullframes,
                        coef,
                        num_iter = 5,
                        neighbor_range = 2,
                        search_range = 5,
                        losstype = 'MSE'):
    
    forflow = smoothflow_withVreg(fullframes = fullframes,
                                  coef = coef,
                                  neighbor_range = neighbor_range,
                                  search_range = search_range,
                                  losstype = losstype)
    backflow = smoothflow_withVreg(fullframes = fullframes[::-1],
                                   coef = coef,
                                   neighbor_range = neighbor_range,
                                   search_range = search_range,
                                   losstype = losstype)
    return forflow, backflow
