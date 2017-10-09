# coding:utf-8

import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

from tqdm import tqdm

def PixelInterp(preframe, postframe, pixelflow, fineness):

    frame_size = preframe.shape[0]
    flow_size = pixelflow.shape[0]

    rem = (frame_size - flow_size)/2
    rem = np.int(rem)
    # pixelflow shape(H, W, 2)
    pixelflow = pixelflow.astype(int)

    # flow_start : pixel value, each pixels in preframe start
    flow_start = preframe[rem:frame_size-rem, rem:frame_size-rem]

    # flow_end : pixel value, each pixels in preframe move to 
    flow_end = np.array([[postframe[y+flow[1], x+flow[0]]\
                          for flow, x in zip(flows,
                                             np.arange(rem, frame_size-rem))]\
                         for flows, y in zip(pixelflow,
                                             np.arange(rem, frame_size-rem))])

    X, Y = np.meshgrid(range(flow_size), range(flow_size))

    # 補間処理
    
    result = np.zeros((fineness, flow_size, flow_size))

    for f in np.arange(fineness):

        X_flowed = X + (f/fineness) * pixelflow[:,:,0]
        Y_flowed = Y + (f/fineness) * pixelflow[:,:,1]

        p_flowed = (1-f/fineness) * flow_start + f/fineness * flow_end

        for y in np.arange(flow_size):
            for x in np.arange(flow_size):

                # 対象の点(x, y)からのそれぞれの距離
                distmap = np.sqrt((X_flowed - x)**2 + (Y_flowed - y)**2)

                p_flowedseries = p_flowed.reshape((flow_size**2))
                distseries = distmap.reshape((flow_size**2))

                idx_near = np.argsort(distseries)[:4]
                p_near = p_flowedseries[idx_near]
                dist_near = distseries[idx_near] + 1e-3
                
                w = np.prod(dist_near)/dist_near
                weight = w/np.sum(w)

                p_weighted = np.sum(weight * p_near)

                result[f, y, x] = p_weighted

                # pdb.set_trace()

    return result


# new method for smooth flow 10/7 added

def bidirect_wrapper(args):
    bidirect_interp(*args)
    
def bidirect_interp(preframe, postframe,
                    forflow, backflow, fineness):
    tar_size, _, _ = forflow.shape
    frame_size, _ = preframe.shape
    rem = int((frame_size - tar_size)/2)
    
    for_interp = temp_interp(start_frame = preframe, end_frame = postframe,
                             flow = forflow, fineness = fineness)
    back_interp = temp_interp(start_frame = postframe, end_frame = preframe,
                              flow = -backflow, fineness = fineness)

    result = np.zeros((fineness, tar_size, tar_size))
    result[0] = preframe[rem:tar_size+rem, rem:tar_size+rem]
    res = np.array([(1-f/fineness)*for_interp[f] + f/fineness*back_interp[-f]
                    for f in np.arange(1, fineness)])
    result[1:] = res

    return result

def temp_interp(start_frame, end_frame, flow, fineness):

    tar_size, _, _ = flow.shape
    frame_size, _ = start_frame.shape
    rem = int((frame_size - tar_size)/2)

    # end slice (spatial) interpolation
    start_p = start_frame[rem:frame_size-rem, rem:frame_size-rem]
    X, Y = np.meshgrid(np.arange(tar_size), np.arange(tar_size))
    end_X = X + flow[:, :, 0]
    end_Y = Y + flow[:, :, 1]
    end_p = space_interp(target_X = end_X, target_Y = end_Y,
                         given_p = end_frame)

    # generate inter-slices shape(fineness, tar_size, tar_size)
    result = np.array([space_interp(
        target_X = X, target_Y = Y,
        given_p = (1-f/fineness)*start_p + f/fineness*end_p,
        given_X = X + (f/fineness) * flow[:, :, 0],
        given_Y = Y + (f/fineness) * flow[:, :, 1])
                       for f in np.arange(fineness)])
    
    return result

def space_interp(target_X, target_Y, # target position where shade p to be calculated
                 given_p, given_X = None, given_Y = None): # fixed shade, and position
    
    tar_size, _ = target_X.shape
    given_size, _ = given_p.shape
    rem = int((given_size - tar_size)/2)
    if given_X is None and given_Y is None: # end-slice shape(40, 40) processing
        given_range = np.arange(-rem, given_size-rem)
        given_X, given_Y = np.meshgrid(given_range, given_range)
    # else:interpolation process, given_p shape(30, 30)
    
    target_p = np.array([[_s_interp(tar_x, tar_y,
                                    given_p, given_X, given_Y)
                          for tar_x, tar_y in zip(tar_xx, tar_yy)]
                         for tar_xx, tar_yy in zip(target_X, target_Y)])
    return target_p

def _s_interp(tar_x, tar_y,
              given_p, given_X, given_Y):
    
    distmap = np.sqrt((given_X - tar_x)**2 + (given_Y - tar_y)**2)

    given_p_flat = given_p.flatten()
    dist_flat = distmap.flatten()

    idx_near = np.argsort(dist_flat)[:4]
    given_p_near = given_p_flat[idx_near]
    dist_near = dist_flat[idx_near] + 1e-3

    w = np.prod(dist_near)/dist_near
    weight = w/np.sum(w)

    p_weighted = np.sum(weight * given_p_near)

    return p_weighted
    
    


