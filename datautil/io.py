import pickle
import os
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
