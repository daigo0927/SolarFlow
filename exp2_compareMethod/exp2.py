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

from SolarFlow import SolarFlow, easySolarFlow
from optimize import opt_hyper
from misc.visualize import draw_cloud
from misc.utils import LinearInterp
from datautil.io import load_pickles

from multiprocessing import Pool


def _wrapper(attr):
    return _process(*attr)

def _process(pkldir, date, region_name, limit_frame):

    print('processing data in {} start'.format(pkldir))

    data = load_pickles(pkldir)
    
    f_origin, w_origin, _ = data.shape

    # train data, fall every every two slices
    if limit_frame%2 == 0:
        raise ValueError('limitframes must be odd number')
    for key in ['shade', 'crop', 'outer']:
        data[key] = np.array(data[key])[:limit_frame]
    

    # experiment setting
    limit_frame = int(limit_frame)
    s_range = int(7)
    n_range = int(2)
    fine = int(2)

    result = OrderedDict()

    # linear interp : shape(217, 26, 26)
    print('linear interpolating ...')
    lin = LinearInterp(data = data_train, crop = s_range)
    lin.interp(fineness = fine)
    result['linear'] = lin.result
    
    
    # proposed method 1, temporal smoothing : shape(217, 23?, 23?)
    print('temporal smoothed interpolating ...')
    t_smooth = SmoothInterp(data = data_train,
                            search_range = s_range,
                            neighbor_range = n_range)
    t_smooth.interp(frameset_size = 3, # <- temporal smoothing
                    space_smooth_value = 10, fineness = fine)
    result['temp_smooth'] = t_smooth.result

    # proposed method 2, spatial smoothing, and double smoothing : shape(217, 23?, 23?)
    s_smooth_values = np.array([6., 3., 1.5])
    labels = ['weak', 'middle', 'strong']
    for s_value, label in zip(s_smooth_values, labels):
        s_smooth = SmoothInterp(data = data_train,
                                search_range = s_range,
                                neighbor_range = n_range)
        print('{} spatial smoothed interpolating ...'.format(label))
        s_smooth.interp(frameset_size = 2,
                        space_smooth_value = s_value, fineness = fine)
        result['space_smooth_{}'.format(label)] = s_smooth.result

        print('{} double smoothed interpolating ...'.format(label))
        s_smooth.interp(frameset_size = 3,
                        space_smooth_value = s_value, fineness = fine)
        result['double_smooth_{}'.format(label)] = s_smooth.result

    # confirm interpolated slices
    f_min, w_min, _ = result['temp_smooth'].shape
    f_test_idx = np.arange(1, f_min, 2)
    error = np.zeros((len(f_test_idx), len(result.keys())))
    colnames = []
    print('interpolation results testing ...')
    for i, key in enumerate(result.keys()):
        res = result[key]
        f, w, _ = res.shape
        drop = int((w - w_min)/2)
        drop_origin = int((w_origin - w_min)/2)
        err = data[f_test_idx,
                   drop_origin:w_min+drop_origin,
                   drop_origin:w_min+drop_origin]\
              - res[f_test_idx, drop:w_min+drop, drop:w_min+drop]
        err = np.mean(err**2, axis = (1,2)) # shape(f_min, )
        
        error[:, i] = err
        colnames.append(key)

    error = pd.DataFrame(error)
    error.columns = colnames
    print(error.describe())

    return [region_name+date, error]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, required = True,
                        help = '/path/to/row_data, contains yyyy-mm-dd')
    parser.add_argument('--date', type = str, nargs = '+', required = True,
                        help = 'objective date, must be [yyyy-mm-dd]')
    parser.add_argument('--region_name', type = str, required = True,
                        help = 'region name, under /row_data/yyyy-mm-dd/pickles, like Tokyo')
    parser.add_argument('--limit_frame', type = int, default = 999,
                        help = 'number of frames for utilize, default [-1](all)')
    args = parser.parse_args()

    pkldirs = [args.data_dir + '/' + d + '/pickles/' + args.region_name\
               for d in args.date]
    attrs = [(pkldir, d, args.region_name,
              args.limit_frame) for pkldir, d in zip(pkldirs, args.date)]

    result = OrderedDict()
    num_cores = int(input('input utilize core number : '))
    pool = Pool(num_cores)
    result['error'] = list(pool.map(_wrapper, attrs))
    pool.close()

    result['config'] = args

    with open('./result.pkl', 'wb') as f:
        pickle.dump(result, f)
            
if __name__ == '__main__':
    main()
