# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

from multiprocessing import Pool

from PixelFlow import PixelFlow

class SolarFlow:

    def __init__(data,
                 SearchRange = 5,
                 NeighborRange = 2):

        self.data = data
        self.Srange = SearchRange
        self.Nrange = NeighborRange

        self.FramePairs = np.array([[self.data[i], self.data[i+1]]\
                                    for i in range(len(self.data)-1)])

        self.flows = None

        self.result = None

    def flow():

        attr = [[pair[0], pair[1], self.Srange, self.Nrange]\
                for pair in self.FramePairs]

        NumCore = np.int(input('input core number : '))
        pool = Pool(NumCore)

        self.flows = list(pool.map(pflow, attr))

    def interp(fineness = 15):

        attr = [[pair[0], pair[1], flow] \
                for pair, flow in zip(self.FramePairs, self.flows)]

        NumCore = np.int(input('input core number : '))
        pool = Pool(NumCore)

        # 渡す処理

        
        

        

        
def pflow(preframe_postframe_Srange_Nrange):

    preframe, postframe, Srange, Nrange = preframe_postframe_Srange_Nrange

    flow = PixelFlow(preframe = preframe,
                     postframe = postframe,
                     SearchRange = Srange,
                     NeighborRange = Nrange)

    return flow

def pinterp(preframe_postframe_pixelflow):

    preframe, postframe, pixelflow = preframe_postframe_pixelflow

    interpolated = PixelInterp(preframe = preframe,
                               postframe = postframe,
                               pixelflow = pixelflow)

    return interpolated


def PixelInterp(preframe, postframe, pixelflow):

    frame_size = preframe.shape[0]
    flow_size = pixelflow.shape[0]

    rem = (frame_size - flow_size)/2
    rem = np.int(rem)
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
