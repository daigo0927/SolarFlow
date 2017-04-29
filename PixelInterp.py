# coding:utf-8

import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

from multiprocessing import Pool

from PixelFlow import PixelFlow

class SolarFlow:

    def __init__(self,
                 data,
                 SearchRange = 5,
                 NeighborRange = 2):

        self.data = data
        self.Srange = SearchRange
        self.Nrange = NeighborRange

        # height/weight of data
        self.size_origin = data.shape[1]
        self.size_result = self.size_origin - 2*self.Srange

        self.FramePairs = np.array([[self.data[i], self.data[i+1]]\
                                    for i in range(len(self.data)-1)])

        self.pool = None

        self.flows = None

        self.result = None

    def flow(self):

        attr = [[pair[0], pair[1], self.Srange, self.Nrange]\
                for pair in self.FramePairs]

        self.flows = list(self.pool.map(pflow, attr))

    def interp(self, fineness = 15):

        NumCore = np.int(input('input core number : '))
        self.pool = Pool(NumCore)

        print('compute pixel flow ...')
        self.flow()

        attr = [[pair[0], pair[1], flow, fineness] \
                for pair, flow in zip(self.FramePairs, self.flows)]

        print('interpolating ...')
        result = np.array(self.pool.map(pinterp, attr))
        result = result.reshape(result.shape[0] * result.shape[1],
                                result.shape[2],
                                result.shape[3])

        # add final frame
        finalframe = self.data[-1,
                               self.Srange:self.size_origin-self.Srange,
                               self.Srange:self.size_origin-self.Srange]
        finalframe = finalframe.reshape(1, self.size_result, self.size_result)
        
        self.result = np.concatenate((result, finalframe), axis = 0)

    def savefig(self, path):
        pass
        

        
def pflow(preframe_postframe_Srange_Nrange):

    preframe, postframe, Srange, Nrange = preframe_postframe_Srange_Nrange

    flow = PixelFlow(preframe = preframe,
                     postframe = postframe,
                     SearchRange = Srange,
                     NeighborRange = Nrange)

    return flow

def pinterp(preframe_postframe_pixelflow_fineness):

    preframe, postframe, pixelflow, fineness = \
                            preframe_postframe_pixelflow_fineness

    interpolated = PixelInterp(preframe = preframe,
                               postframe = postframe,
                               pixelflow = pixelflow,
                               fineness = fineness)

    return interpolated


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
    
