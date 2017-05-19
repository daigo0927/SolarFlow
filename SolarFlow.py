# coding:utf-8

import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

from multiprocessing import Pool
from tqdm import tqdm

from PixelFlow import PixelFlow
from PixelInterp import PixelInterp

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
        self.doubleflows = None

        self.result = None

    def flow(self):

        attr = [[pair[0], pair[1], self.Srange, self.Nrange]\
                for pair in self.FramePairs]

        if self.pool == 0:
            self.flows = list([pflow(att) for att in tqdm(attr)])
        else:
            self.flows = list(self.pool.map(pflow, attr))

    def doubleflow(self):

        attr = [[pair[0], pair[1], self.Srange, self.Nrange]\
                for pair in self.FramePairs]

        if self.pool == 0:
            self.doubleflows = list([biflow(att) for att in tqdm(attr)])
        else:
            self.doubleflows = list(self.pool.map(biflow, attr))
        

    def interp(self, fineness = 15, method = 'for'):

        NumCore = np.int(input('input core number : '))
        if NumCore == 0:
            self.pool = 0
        else:
            self.pool = Pool(NumCore)

        print('compute pixel flow ...')

        if method == 'for':
            self.flow()

            attr = [[pair[0], pair[1], flow, fineness] \
                    for pair, flow in zip(self.FramePairs, self.flows)]

            print('interpolating ...')
            if self.pool == 0:
                result = np.array([pinterp(att) for att in tqdm(attr)])
            else:
                result = np.array(self.pool.map(pinterp, attr))

        elif method == 'bi':
            self.doubleflow()

            attr = [[pair[0], pair[1], dflow[0], dflow[1], fineness] \
                    for pair, dflow in zip(self.FramePairs, self.doubleflows)]

            print('initerpolating ...')
            if self.pool == 0:
                result = np.array([biinterp(att) for att in tqdm(attr)])
            else:
                result = np.array([self.pool.map(biinterp, attr)])
                
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

def biflow(preframe_postframe_Srange_Nrange):

    preframe, postframe, Srange, Nrange = preframe_postframe_Srange_Nrange

    forflow = PixelFlow(preframe = preframe,
                        postframe = postframe,
                        SearchRange = Srange,
                        NeighborRange = Nrange)

    backflow = PixelFlow(preframe = postframe,
                         postframe = preframe,
                         SearchRange = Srange,
                         NeighborRange = Nrange)

    return np.array([forflow, backflow])

    
def biinterp(preframe_postframe_forflow_backflow_fineness):

    preframe, postframe, forflow, backflow, fineness = \
                            preframe_postframe_forflow_backflow_fineness

    interp_for = PixelInterp(preframe = preframe,
                             postframe = postframe,
                             pixelflow = forflow,
                             fineness = fineness)

    interp_back = PixelInterp(preframe = postframe,
                              postframe = preframe,
                              pixelflow = backflow,
                              fineness = fineness)

    interpolated = np.array([interp_for[0]])
    for i in np.arange(fineness)[1:]:
        interp = (1-i/fineness) * interp_for[i] \
                 + i/fineness * interp_back[-i]
        interpolated = np.append(interpolated, np.array([interp]),
                                 axis = 0)

    return interpolated
