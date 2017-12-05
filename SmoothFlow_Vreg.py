# coding: utf-8

import numpy as np
from tqdm import tqdm

# vector field estimation with regularizer with v

# area mathing loss ---------------------------
# receive temporally continuing 2 frames, each shape(frame_size, frame_size)
# return matcing loss field (shape(frame_size-2*s_range, frame_size-2*s_range, 7, 7))
def matching_loss(preframe,
                  postframe,
                  neighbor_range = 2,
                  search_range = 5,
                  losstype = 'MSE'):
    
    frame_size = preframe.shape[0]
    n_range = neighbor_range
    s_range = search_range

    l_match = np.array([[_search(area = preframe[y-n_range:y+n_range+1,
                                                 x-n_range:x+n_range+1],
                                 search_area = postframe[y-s_range:y+s_range+1,
                                                         x-s_range:x+s_range+1],
                                 losstype = losstype)\
                         for x in np.arange(s_range, frame_size - s_range)]\
                        for y in np.arange(s_range, frame_size - s_range)])
    return l_match

# search function for single area
def _search(area, search_area, losstype):
    area_size = area.shape
    search_size = search_area.shape

    ls = np.array([[_match(prearea = area,
                           postarea = search_area[y:y+area_size[0],
                                                  x:x+area_size[1]],
                           losstype = losstype)
                    for x in np.arange(search_size[1] - area_size[1] + 1)]\
                   for y in np.arange(search_size[0] - area_size[0] + 1)])
    return ls

# mathing loss calculation for one-by-one area matching
def _match(prearea, postarea, losstype):
    if losstype == 'MSE':
        l = np.sqrt((postarea - prearea)**2)
    else:
        raise ValueError('losstype {} is not available'.format(losstype))
    return np.mean(l)
# ---------------------------------area mathing loss


# vector extraction from given losses ( shape(n, 7, 7))---------
def get_vector(loss, coef):
    assert loss.shape[0] == len(coef),'args (loss and coef) should have same length, loss {}, coef {}'.format(loss.shape, len(coef))

    # loss summation
    coef = np.reshape(coef, (-1, 1, 1)) # shape(n, 1, 1) for numpy bloadcasting
    loss_sum = np.sum(loss*coef, axis = 0)

    min_idx = np.array([i[0] for i in np.where(loss_sum == np.min(loss_sum))])
    # axis operation for converting a flow vector
    min_idx = min_idx[::-1] - (np.array(loss_sum.shape) - 1.)/2
    
    return min_idx # array([v_x, v_y])
# --------------------------------------------------------- vector extraction


# spatial regularizer -------------------------------------
# receive vector at target position, h*w field of flow vector (shape(h, w, 2)),
# and expecting vector value range
# return spatial vector regularizer (shape(vec_range, vec_range))
def reg_spatial(vec, vec_field, vec_range):
    # mean vector of surrounding 8 vectors
    h, w, _ = vec_field.shape
    vec_ = (np.sum(vec_field, axis = (0, 1)) - vec)/(h*w-1)

    vec_radius = int((vec_range-1)/2)
    vecs_x, vecs_y = np.meshgrid(np.arange(-vec_radius, vec_radius+1),
                                 np.arange(-vec_radius, vec_radius+1))
    vecs_x = np.reshape(vecs_x, (vec_range, vec_range, 1))
    vecs_y = np.reshape(vecs_y, (vec_range, vec_range, 1))
    vecs = np.concatenate([vecs_x, vecs_y], axis = 2) # expecting vectors

    vec_ = np.reshape(vec_, (1, 1, 2)) # reshape for bloadcasting
    # calculate regularize value (l2 norm, just now) for each expecting vector
    regs = np.sqrt(np.sum((vec_ - vecs)**2, axis = 2))

    return regs # shape(7, 7)
# ---------------------------------------- spatial regularizer


# temporal regularizer ------------------------------------------
# receive  vector at target position, 3 length flow series (shape(3, 2)),
# and expecting vector value range
# return temporal vector regularizer (shape(vec_range, vec_range))
def reg_temporal(vec, vec_series, vec_range):
    # mean vector of pre&post 2 vectors
    t, _ = vec_series.shape
    vec_ = (np.sum(vec_series, axis = 0) - vec)/(t-1)

    vec_radius = int((vec_range-1)/2)
    vecs_x, vecs_y = np.meshgrid(np.arange(-vec_radius, vec_radius+1),
                                 np.arange(-vec_radius, vec_radius+1))
    vecs_x = np.reshape(vecs_x, (vec_range, vec_range, 1))
    vecs_y = np.reshape(vecs_y, (vec_range, vec_range, 1))
    vecs = np.concatenate([vecs_x, vecs_y], axis = 2) # expecting vectors

    vec_ = np.reshape(vec_, (1, 1, 2)) # reshape for bloadcasting
    # calculate regularize value (l2 norm, just now) for each expecting vector
    regs = np.sqrt(np.sum((vec_ - vecs)**2, axis = 2))

    return regs # shape(7, 7)
# ------------------------------------------ temporal regularizer


# temp-spatial regularizer -------------------------------
# receive post frame vector field (shape(vec_range, vec_range, 2))
# return temp-spatial regularizer (shape(vec_range, vec_range))
def reg_tempspatial(post_vec_field):
    vec_range, _, _ = post_vec_field.shape
    vec_radius = int((vec_range-1)/2)

    vecs_x, vecs_y = np.meshgrid(np.arange(-vec_radius, vec_radius+1),
                                 np.arange(-vec_radius, vec_radius+1))
    vecs_x = np.reshape(vecs_x, (vec_range, vec_range, 1))
    vecs_y = np.reshape(vecs_y, (vec_range, vec_range, 1))
    vecs = np.concatenate([vecs_x, vecs_y], axis = 2) # expecting vectors

    regs = np.sqrt(np.sum((post_vec_field - vecs)**2, axis = 2))

    return regs # shape(vec_range, vec_range)
# -------------------------------------- temp-spatial regularizer

# calculate all regularizer ------------------------------
# from given (3-D) vector field (shape(num_loss, frame_size, frame_size, 2))
# return each reagularizers (shape(num_loss, frame_size, frame_size, vec_range, vec_range))
def reg_all(vec_field, vec_range):
    num_loss, frame_size, _, _ = vec_field.shape

    # spatial term, shape(num_loss, frame_size, frame_size, vec_range, vec_range)
    reg_s = np.array([[[reg_spatial(vec = vec_field[t, y, x],
                                    vec_field = vec_field[t,
                                                          max(y-1,0):y+2,
                                                          max(x-1,0):x+2],
                                    vec_range = vec_range)
                        for x in np.arange(frame_size)]\
                       for y in np.arange(frame_size)]\
                      for t in np.arange(num_loss)])

    # temporal term, shape(num_loss, frame_size, frame_size, vec_range, vec_range)
    reg_t = np.array([[[reg_temporal(vec = vec_field[t, y, x],
                                     vec_series = vec_field[max(t-1,0):t+2, y, x],
                                     vec_range = vec_range)
                        for x in np.arange(frame_size)]\
                       for y in np.arange(frame_size)]\
                      for t in np.arange(num_loss)])

    # temporal-spatial term
    vec_radius = int((vec_range-1)/2)
    # shape(num_loss-1, frame_size-2*vec_range, frame_size-2*vec_range, vec_range, vec_range)
    reg_ts_ = np.array([[[reg_tempspatial(post_vec_field\
                                          = vec_field[t+1,
                                                      y-vec_radius:y+vec_radius+1,
                                                      x-vec_radius:x+vec_radius+1])
                          for x in np.arange(vec_radius, frame_size-vec_radius)]\
                         for y in np.arange(vec_radius, frame_size-vec_radius)]\
                        for t in np.arange(num_loss-1)])
    reg_ts = np.zeros_like(reg_s) # zero padding operation for preparing same shape regs
    reg_ts[:num_loss-1,
           vec_radius:frame_size-vec_radius,
           vec_radius:frame_size-vec_radius] = reg_ts_

    return reg_s, reg_t, reg_ts
# ------------------------------------------------------ all regularizer
    

# estimate vector field optimizing matching-loss and regularizers ---------------------
class VectorOptimizer(object):

    def __init__(self,
                 fullframes,
                 neighbor_range = 2,
                 search_range = 5,
                 losstype = 'MSE'):
        
        self.fullframes = fullframes
        self.n_range = neighbor_range
        self.s_range = search_range
        self.losstype = losstype

        # initial vector calcultion by area matching loss
        print('calculate matching-loss, and initialize vector field')
        self._match_loss()
        self._vec_init()

        self.coef = None
        self.loss_concat = None
        
    def _vec_init(self):
        self.vec_field = np.array([[[get_vector(l, coef = np.array([1]))
                                     for l in loss]\
                                    for loss in loss_frame]\
                                   for loss_frame in self.match_loss])
                
    def _match_loss(self):
        self.num_frames, self.frame_size_origin, _ = self.fullframes.shape
        match_loss = np.array([matching_loss(preframe = pre,
                                             postframe = post,
                                             neighbor_range = self.n_range,
                                             search_range = self.s_range,
                                             losstype = self.losstype)
                               for pre, post in zip(self.fullframes[:self.num_frames-1],
                                                    self.fullframes[1:])])
        self.num_loss, self.frame_size, _, self.vec_range, _ = match_loss.shape
        # reshape for bloadcasting
        self.match_loss = np.reshape(match_loss,
                                     (self.num_loss,
                                      self.frame_size, self.frame_size,
                                      1, self.vec_range, self.vec_range))

    def optimize(self, num_iter, coef):
        self.coef = coef
        print('optimize vector field ...')
        for i in tqdm(range(num_iter)):
            reg_s, reg_t, reg_ts = reg_all(self.vec_field, self.vec_range)
            # concatenate matching-loss and regularizers
            self.loss_concat = self._concat(reg_s, reg_t, reg_ts)
            self.vec_field = np.array([[[get_vector(l, coef = coef)
                                         for l in loss]\
                                        for loss in loss_frame]\
                                       for loss_frame in self.loss_concat])
        return self.vec_field, self.loss
            
    @property
    def loss(self):
        coef = np.reshape(self.coef, (1, 1, 1, -1, 1, 1))
        loss_field = np.min(np.sum(coef*self.loss_concat, axis = 3),
                            axis = (3, 4))
        return loss_field

    def _concat(self, reg_s, reg_t, reg_ts):
        reg_s = np.reshape(reg_s,
                           (self.num_loss,
                            self.frame_size, self.frame_size,
                            1, self.vec_range, self.vec_range))
        reg_t = np.reshape(reg_t,
                           (self.num_loss,
                            self.frame_size, self.frame_size,
                            1, self.vec_range, self.vec_range))
        reg_ts = np.reshape(reg_ts,
                           (self.num_loss,
                            self.frame_size, self.frame_size,
                            1, self.vec_range, self.vec_range))

        loss_concat = np.concatenate([self.match_loss, reg_s, reg_t, reg_ts],
                                      axis = 3)
        return loss_concat
        

def smoothflow_withVreg(fullframes,
                        coef,
                        num_iter = 5,
                        neighbor_range = 2,
                        search_range = 5,
                        losstype = 'MSE'):
    
    num_frames, frame_size_origin, _ = fullframes.shape
    n_range = neighbor_range
    s_range = search_range

    optimizer = VectorOptimizer(fullframes = fullframes,
                                neighbor_range = n_range,
                                search_range = s_range,
                                losstype = losstype)
    optimizer.optimize(num_iter, coef)

    return optimizer.vec_field
