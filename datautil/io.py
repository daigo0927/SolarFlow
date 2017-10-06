import pickle
import os
import numpy as np
from glob import glob

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
        
