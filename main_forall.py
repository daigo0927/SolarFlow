# coding:utf-8

import os, sys
import numpy as np
import pickle
import argparse
import pdb

from tqdm import tqdm

from SolarFlow import SolarFlow, easySolarFlow
from misc.visualize import draw_cloud
from misc.utils import LinearInterp

from multiprocessing import Pool


def _wrapper(attr):
    _process(*attr)

def _process(pkldir, limit_frame, search_range, neighbor_range, method, fine):

    print('processing data in {} start'.format(pkldir))
    
    with open(pkldir + '/shade.pkl', 'rb') as f:
        data = pickle.load(f)
    limit_frame = int(limit_frame)
    s_range = int(search_range)
    n_range = int(neighbor_range)

    if limit_frame > 0:
        print('interp only data[:{}]'.format(limit_frame))
        data = data[:int(limit_frame)]

    easyflow = easySolarFlow(data = data,
                             search_range = s_range,
                             neighbor_range = n_range)

    if method == 'linear':
        lin = LinearInterp(data = data, crop = s_range)
        lin.interp(fineness = fine)
        easyflow.result = lin.result
    else:
        easyflow.interp(fineness = fine, method = method)

    shade_fine = easyflow.result

    with open(pkldir + '/outer_fine.pkl', 'rb') as f:
        outer_fine = pickle.load(f)

    _, shade_size, _ = shade_fine.shape
    drop = int(outer_fine.shape[1] - shade_size)
    total_fine = np.array(outer_fine)[:, drop:drop+shade_size, drop:drop+shade_size]\
                 *(1. - shade_fine)
    print('save fine shade and total radiation')
    with open(pkldir + '/shade_fine.pkl', 'wb') as f:
        pickle.dump(shade_fine, f)
    with open(pkldir + '/total_fine.pkl', 'wb') as f:
        pickle.dump(total_fine, f)

    print('interpolation of data in {} completed'.format(pkldir))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, required = True,
                        help = '/path/to/row_data, contains yyyy-mm-dd')
    parser.add_argument('--date', type = str, nargs = '+', required = True,
                        help = 'objective date, must be [yyyy-mm-dd]')
    parser.add_argument('--region_name', type = str, required = True,
                        help = 'region name, under /row_data/yyyy-mm-dd/pickles, like Tokyo')
    parser.add_argument('--limit_frame', type = int, default = -1,
                        help = 'number of frames for utilize, default [-1](all)')
    parser.add_argument('--s_range', type = int, default = 5,
                        help = 'search range for pixel flow, default [5]')
    parser.add_argument('--n_range', type = int, default = 2,
                        help = 'neighrbor range for pixel flow, default [2]')
    parser.add_argument('--method', type = str,
                        choices = ['linear', 'for', 'bi'], required = True,
                        help = 'interpolation mode, choose from [linear, for, bi]')
    parser.add_argument('--fine', type = int, default = 15,
                        help = 'interpolation fineness, default [15]')
    args = parser.parse_args()

    pkldirs = [args.data_dir + '/' + d + '/pickles/' + args.region_name\
               for d in args.date]
    attrs = [(pkldir, args.limit_frame, args.s_range, args.n_range,
              args.method, args.fine) for pkldir in pkldirs]
    
    num_cores = int(input('input utilize core number : '))
    pool = Pool(num_cores)
    pool.map(_wrapper, attrs)
    pool.close()

            
if __name__ == '__main__':
    main()
