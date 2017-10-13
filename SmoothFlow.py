import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from multiprocessing import Pool
from tqdm import tqdm

from PixelFlow import visFlow, GetPixelFlow
from PixelInterp import bidirect_interp, bidirect_wrapper

class SmoothInterp(object):

    def __init__(self,
                 data,
                 search_range = 5, neighbor_range = 2,
                 multiprocess = False):
        self.data = data
        self.s_range = search_range
        self.n_range = neighbor_range
        self.num_frames = len(data)

        if multiprocess:
            num_cores = int(input('input utilize core number : '))
            self.pool = Pool(num_cores)
        else:
            self.pool = None

        self.fflow = None
        self.bflow = None

        self.result = None

    def interp(self,
               frameset_size = 2, space_smooth_value = 1.5,
               fineness = 15):
        
        self.fflow, self.bflow \
            = GetPixelFlow(all_frames = self.data,
                           frameset_size = frameset_size,
                           search_range = self.s_range,
                           neighbor_range = self.n_range,
                           space_smooth_value = space_smooth_value)

        args = [(self.data[f], self.data[f+1],
                 self.fflow[f], self.bflow[f], fineness)
                for f in np.arange(self.num_frames - 1)]

        if self.pool is not None:
            result = np.array(self.pool.map(bidirect_wrapper, args))
        else:
            result = np.array([bidirect_interp(*arg) for arg in tqdm(args)])

        # num_sets, fineness, width, width
        s, f, w, _ = result.shape
        result = np.reshape(result, (s*f, w, w))
        
        # add final frame
        finframe = self.data[-1,
                             self.s_range : w+self.s_range,
                             self.s_range : w+self.s_range]
        finframe = np.array([finframe])
        self.result = np.concatenate((result, finframe), axis = 0)
