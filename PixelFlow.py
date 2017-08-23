# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

# take pre/post frame
# return pixel flow: shape(ygrid, xgrid, [flow_x,flow_y])
def PixelFlow(preframe,
              postframe,
              SearchRange = 5, # range for search pixel (radius)
              NeighborRange = 2): # range for treat as neightbor (radius)

    f_size = preframe.shape[0]

    s_range = SearchRange
    n_range = NeighborRange
    
    pixelflow = np.array([[PixelSearch(\
                            SearchArea = postframe[y-s_range:y+s_range+1,
                                                   x-s_range:x+s_range+1],
                            pixels = preframe[y-n_range:y+n_range+1,
                                              x-n_range:x+n_range+1])\
                           - np.array([s_range, s_range]) \
                           for x in np.arange(s_range, f_size-s_range)]\
                          for y in np.arange(s_range, f_size-s_range)])

    return pixelflow
    


def PixelSearch(SearchArea, # area for search flow
                pixels): # objective pixel, and surrounding pixels

    S_size = SearchArea.shape
    p_size = np.array(pixels.shape)
    
    loss = np.array([[PixelLoss(PostArea = SearchArea[y:y+p_size[0],
                                                      x:x+p_size[1]],
                                PreArea = pixels,
                                losstype = 'MSE')\
                      for x in range(S_size[1] - p_size[1] + 1)]\
                     for y in range(S_size[0] - p_size[0] + 1)])
    
    min_idx = np.array([i[0] for i in np.where(loss == np.min(loss))])
    # return center position of the pixels
    min_idx = min_idx[::-1] + (p_size-1.)/2
    # return pixel movement ([xflow, yflow])
    return min_idx

        

# compare 2 areas
def PixelLoss(PostArea, PreArea, losstype = 'MSE'):

    if losstype == 'MSE':
        loss = np.sqrt((PostArea - PreArea)**2)
    elif loss == 'other':
        # other loss function
        pass

    return np.mean(loss)


# visualize pixel flow
# pixelflow shape(y_len, x_len, 2)
# pixelflow contents : [xflow, yflow]
# !!! sort axis as sns.heatmap, top-left:(0,0)
def visFlow(pixelflow, vismargin=5):

    y_len, x_len, _ = pixelflow.shape
    X, Y = np.meshgrid(range(x_len), range(y_len))

    fig = sns.plt.figure()
    ax = fig.add_subplot(111)
    ax.quiver(X, Y, pixelflow[:,:,0], -pixelflow[:,:,1],
              facecolor = 'blue')
    ax.set_xlim([-vismargin, x_len+vismargin])
    ax.set_ylim([y_len+vismargin, -vismargin])


# new methods 8/20 -

def _get_loss(search_area, obj_area):

    search_size = search_area.shape
    obj_size = obj_area.shape

    loss = np.array([[PixelLoss(PostArea = search_area[y:y+obj_size[0],
                                                       x:x+obj_size[1]],
                                PreArea = obj_area,
                                losstype = 'MSE')\
                      for x in range(search_size[1] - obj_size[1] + 1)]\
                     for y in range(search_size[0] - obj_size[0] + 1)])
    return loss

def _get_lossmap(preframe,
                 postframe,
                 search_range = 5,
                 neighbor_range = 2):

    frame_size = preframe.shape[0]
    s_range = search_range
    n_range = neighbor_range

    lossmap = np.array([[_get_loss(search_area = postframe[y-s_range:y+s_range+1,
                                                           x-s_range:x+s_range+1],
                                   obj_area = preframe[y-n_range:y+n_range+1,
                                                       x-n_range:x+n_range+1])\
                         for x in np.arange(s_range, frame_size - s_range)]\
                        for y in np.arange(s_range, frame_size - s_range)])

    return lossmap

def _det_flow(lossmaps, preflow = None):
    
    num_frames, map_size, _, flow_range, _ = lossmaps.shape
    flow_radius = int((flow_range-1)/2)
    assert num_frames <= 2, 'the case num_frames>2 is not supported yet'

    if num_frames == 1:
        cumulative_lossmap = lossmaps[0]
    else:
        cumulative_lossmap \
            = np.array([[_temp_smoother(preloss = lossmaps[0, y, x],
                                        postloss = lossmaps[1],
                                        pre_xy = np.array([x, y]))\
                         for x in np.arange(flow_radius, map_size-flow_radius)]\
                        for y in np.arange(flow_radius, map_size-flow_radius)])
        
    # cumulative_lossmap: shape(30/24, 30/24, 7, 7)
    # determine pixelflow
    if preflow is None:
        flow = np.array([[_det_f(ll) for ll in l]\
                         for l in cumulative_lossmap])
    else:
        assert cumulative_lossmap.shape[:2] == preflow.shape[:2],\
            'preflow shape error'
        flow = np.array([[_det_f(ll, pre_f = pre_ff)\
                          for ll, pre_ff in zip(l, pre_f)]\
                         for l, pre_f in zip(cumulative_lossmap, preflow)])
    return flow

def _det_f(loss, pre_f = None):

    if pre_f is not None:
        pre_f_shift = pre_f + (np.array(loss.shape) - 1.)/2
        pre_f_shift = np.round(pre_f_shift).astype(int)
        loss_limit = loss[max(pre_f_shift[1]-1, 0):pre_f_shift[1]+2,
                          max(pre_f_shift[0]-1, 0):pre_f_shift[0]+2]
        min_idx = np.array([i[0] for i in np.where(loss == np.min(loss_limit))])
    else:
        min_idx = np.array([i[0] for i in np.where(loss == np.min(loss))])
        
    min_idx = min_idx[::-1] - (np.array(loss.shape) - 1.)/2
    # return xy_position at
    return min_idx

def _temp_smoother(preloss, postloss, pre_xy):
    # preloss:(7, 7), postloss:(30, 30, 7, 7), xy_origin:(2,)
    flow_range, _ = preloss.shape
    flow_radius = int((flow_range-1)/2)
    cumulative_loss = np.array([[preloss[y, x] \
                                 + np.min(postloss[pre_xy[1]+(y-flow_radius),
                                                   pre_xy[0]+(x-flow_radius),
                                                   max(y-1, 0):y+2,
                                                   max(x-1, 0):x+2])\
                                 for x in np.arange(flow_range)]\
                                for y in np.arange(flow_range)])
    return cumulative_loss # shape(flow_range, flow_range)
    
def get_flow(frames,
             search_range = 5,
             neighbor_range = 2,
             init = True,
             preflow = None):

    if not init:
        assert preflow is not None, 'need pre-pixelflow for temporal consistency'

    lossmaps = np.array([_get_lossmap(preframe = pre,
                                      postframe = post,
                                      search_range = search_range,
                                      neighbor_range = neighbor_range)\
                         for pre, post in zip(frames[:-1], frames[1:])])

    flow = _det_flow(lossmaps, preflow = preflow)
    return flow

def _space_smoother(flow, theta = 2.5): # theta:0-8

    converge = False
    while not converge:
        smoothed_flow = np.array([[_s_smoother(flow = flow[max(y-1,0):y+2,
                                                           max(x-1,0):x+2],
                                                flow_obj = flow[y,x],
                                                theta = theta)\
                                    for x in np.arange(flow.shape[1])]\
                                   for y in np.arange(flow.shape[0])])
        flow = smoothed_flow[:, :, :2]
        if not False in smoothed_flow[:, :, 2]:
            converge = True

    return flow

def _s_smoother(flow, flow_obj, theta):
    # flow:shape(3,3,2), flow_obj(2)
    flow_avg = (np.sum(flow, axis = (0,1)) - flow_obj)/(flow.shape[0]*flow.shape[1]-1)
    if np.linalg.norm(flow_avg - flow_obj) > theta:
        flow_smooth = np.append(flow_avg, False) # False -> 0
    else:
        flow_smooth = np.append(flow_obj, True) # True -> 1
    return flow_smooth

def get_backflow(forflow):
    
    return backflow


    
    
    

    
    

    


    
    
    
