import pickle
import os
import numpy as np
from glob import glob
import pandas as pd

def load_pickles(path):
    
    pklnames = ['crop', 'total_fine',
                'outer', 'outer_fine',
                'shade', 'shade_fine']
    data = {}
    
    for pkl in pklnames:
        pklpath = path + '/' + pkl + '.pkl'
        if not os.path.exists(pklpath):
            print('path {} not exist, continue'.format(pklpath))
            continue

        with open(pklpath, 'rb') as f:
            data[pkl] = pickle.load(f)

    return data

def pad_gdata(gdata):

    (time_length, _) = gdata.shape
    while time_length%60 != 0:
        gdata = np.vstack((gdata[0], gdata))
        (time_length, _) = gdata.shape
        
    print('returned data shape : {}'.format(gdata.shape))
    return gdata

class gdata_preprocesser(object):

    def __init__(self, gpath):
        gdata_origin_ = np.array(pd.read_csv(gpath, header = None))
        self.gdata_origin = pad_gdata(gdata_origin_)

    def move_avg(self, limit_frame = 217, step = 150):
        fin_gframe = 32400+(limit_frame-1)*150 # mainly 18h frame

        radius = int(step/2)
        gdata = np.array([np.mean(self.gdata_origin[i-radius:i+radius, 1])
                          for i in np.arange(32399, fin_gframe, step)])
        return gdata
        
