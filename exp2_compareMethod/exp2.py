# coding:utf-8

import os, sys
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
import pickle
import argparse
import pdb

from tqdm import tqdm
from collections import OrderedDict

from optimize import Interpolater, croparray
from misc.visualize import draw_cloud
from datautil.io import load_pickles

from multiprocessing import Pool

def _wrapper(attr):
    return _process(*attr)

def _process(pkldir, date, region_name, limit_frame, max_evals, validation):

    print('processing data in {} start'.format(pkldir))
    data = load_pickles(pkldir)
    
    result = OrderedDict()
    # reconstructed global solar radiation
    inter = Interpolater(data = data, limit_frame = limit_frame, validation = validation)
    result['linear'] = inter.linear_interp()
    result['normal'] = inter.flow_interp()
    result['double'], hparams_d = inter.flow_interp_doubleregs(max_evals = max_evals)
    result['triple'], hparams_t = inter.flow_interp_tripleregs(max_evals = max_evals)
    
    # confirm interpolated slices
    limit_frame = min(len(data['crop']), limit_frame)
    train_idx = np.arange(0, limit_frame, validation)
    val_idx = True^np.array([i in train_idx for i in np.arange(limit_frame)])
    error = np.zeros((len(result['triple'][val_idx].flatten()), len(result.keys())))
    colnames = []
    print('interpolation results testing ...')
    for i, (key, res) in enumerate(result.items()):
        crop_ = croparray(inter.data['crop'], result['triple'])
        res_ = croparray(res, result['triple'])
        diff = np.sqrt((crop_ - res_)**2)
                
        error[:, i] = diff[val_idx].flatten()
        colnames.append(key)

    error = pd.DataFrame(error)
    error.columns = colnames
    print(error.describe())

    return {'date':region_name+date, 'error':error,
            'hparams_double':hparams_d, 'hparams_triple':hparams_t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, required = True,
                        help = '/path/to/row_data, contains yyyy-mm-dd')
    parser.add_argument('--date', type = str, nargs = '+', required = True,
                        help = 'objective date, must be [yyyy-mm-dd]')
    parser.add_argument('--region_name', type = str, required = True,
                        help = 'region name, under /row_data/yyyy-mm-dd/pickles, like Tokyo')
    parser.add_argument('--limit_frame', type = int, default = 999,
                        help = 'number of frames for utilize, default [999](all)')
    parser.add_argument('--max_evals', type = int, default = 50,
                        help = 'number of max evaluation in hyperopt [50]')
    parser.add_argument('--validation', type = int, default = 2,
                        help = 'number of validation fold [2]')
    args = parser.parse_args()

    pkldirs = [args.data_dir + '/' + d + '/pickles/' + args.region_name\
               for d in args.date]
    attrs = [(pkldir, d, args.region_name,
              args.limit_frame, args.max_evals, args.validation)
             for pkldir, d in zip(pkldirs, args.date)]

    result = OrderedDict()

    num_cores = int(input('input utilize core number : '))
    if num_cores <= 1:
        result['error'] = [_wrapper(attr) for attr in attrs]
    else:
        pool = Pool(num_cores)
        result['error'] = list(pool.map(_wrapper, attrs))
        pool.close()

    result['config'] = args

    with open('./result.pkl', 'wb') as f:
        pickle.dump(result, f)
            
if __name__ == '__main__':
    main()
