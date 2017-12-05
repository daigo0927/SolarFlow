# coding:utf-8

import numpy as np
from tqdm import tqdm

from SmoothFlow_Vreg import smoothflow_withVreg
from PixelInterp import PixelInterp

def fullinterp(fullframes, vec_forward, vec_backward, fineness):

    num_frames, frame_size_origin, _ = fullframes.shape
    _, frame_size, _, _ = vec_forward.shape
    rem = int((frame_size_origin - frame_size)/2)

    interp_full = np.zeros(((num_frames-1)*fineness+1, frame_size, frame_size))
    for n in tqdm(np.arange(num_frames-1)):
        interp_full[n*fineness:(n+1)*fineness] = _interp(preframe = fullframes[n],
                                                         postframe = fullframes[n+1],
                                                         forflow = vec_forward[n],
                                                         backflow = vec_backward[-n],
                                                         fineness = fineness)
    interp_full[-1] = fullframes[-1, rem:frame_size+rem, rem:frame_size+rem]
    
    return interp_full

def _interp(preframe, postframe,
            forflow, backflow, fineness):
    frame_size_origin, _ = preframe.shape
    frame_size, _, _ = forflow.shape
    rem = int((frame_size_origin - frame_size)/2)
    
    interp_for = PixelInterp(preframe, postframe,
                             forflow, fineness)
    interp_back = PixelInterp(postframe, preframe,
                              backflow, fineness)
    interp_mix = np.zeros_like(interp_for)
    interp_mix[0] = preframe[rem:frame_size+rem, rem:frame_size+rem]
    interp_mix[1:] = np.array([(1-f/fineness)*interp_for[f] + f/fineness*interp_back[-f]
                               for f in np.arange(1, fineness)])
    return interp_mix
