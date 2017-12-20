# coding:utf-8

import os, sys
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
import pickle
import argparse
import pdb

from tqdm import tqdm
from fabric.colors import green
from collections import OrderedDict

from misc.visualize import draw_cloud
from optimize import Interpolater
from datautil.io import load_pickles, gdata_preprocesser

from multiprocessing import Pool


def _wrapper(attr):
    return _process(*attr)

def _process(pkldir, gdata_dir, date, region_name, limit_frame, max_evals):

    print('processing data in {} start'.format(pkldir))

    # oroginal shade ratio
    data = load_pickles(pkldir)
    # original total radiation data, shape(limit_frame, 40, 40)
    limit_frame = min(limit_frame, data['crop'].shape[0])
    tdata = data['crop'][:limit_frame]
    sdata = data['shade'][:limit_frame]
    _, frame_size_origin, _ = tdata.shape
    obj_idx = np.array([19, 20]) # ground measurement location for the 120th building
    # total radiation at objective location, shppe(217, )
    obj_total = tdata[:limit_frame, obj_idx[1], obj_idx[0]]
    # outer radiation at objective location, shape(3241, )
    f_fine = (limit_frame-1)*15+1
    obj_outer_fine = data['outer_fine'][:f_fine, obj_idx[1], obj_idx[0]]

    # load ground data
    gpath = gdata_dir + '/' + ''.join(date.split('-')) + '_1sec.csv'
    processor = gdata_preprocesser(gpath)
    gdata = processor.move_avg(limit_frame, 150)
    gdata_fine = processor.move_avg(limit_frame, 10)

    # experiment setting
    s_range = int(5)
    n_range = int(2)
    fineness = int(15)

    result = OrderedDict()
    inter = Interpolater(data = data, limit_frame = limit_frame, fineness = 15)
    result['linear'] = inter.linear_interp()
    result['normal_MSE'] = inter.flow_interp(losstype = 'MSE')
    result['normal_NCC'] = inter.flow_interp(losstype = 'NCC')
    result['double_MSE'], hparams_d_MSE = inter.flow_interp_doubleregs(losstype = 'MSE',
                                                                       max_evals = max_evals)
    result['double_NCC'], hparams_d_NCC = inter.flow_interp_doubleregs(losstype = 'NCC',
                                                                       max_evals = max_evals)
    result['triple_MSE'], hparams_t_MSE = inter.flow_interp_tripleregs(losstype = 'MSE',
                                                                       max_evals = max_evals)
    result['triple_NCC'], hparams_t_NCC = inter.flow_interp_tripleregs(losstype = 'NCC',
                                                                       max_evals = max_evals)

    # record error
    simular = np.zeros((limit_frame - 1, len(result.keys())))
    colnames = []
    print(green('interpolating results testing by ground measured data'))
    for i, (key, res) in enumerate(result.items()):
        tvalue = pickPointValue(tvalues = res,
                                target_idx_origin = obj_idx,
                                frame_size_origin = frame_size_origin)
        simular[:, i] = np.array([cosSimular(tvalue[i*fineness:(i+1)*fineness],
                                             gdata_fine[i*fineness:(i+1)*fineness])
                                  for i in np.arange(limit_frame - 1)])
        colnames.append(key)

    simular = pd.DataFrame(simular)
    simular.columns = colnames
    print(simular.describe())
        
    return {'date':region_name+date, 'simularity':simular,
            'hparams_d_MSE':hparams_d_MSE, 'hparams_d_NCC':hparams_d_NCC,
            'hparams_t_MSE':hparams_t_MSE, 'hparams_t_NCC':hparams_t_NCC}

# extract target point radiation value, basically from total radiation value
# this func prepared for that interpolated result can be different size from original data
def pickPointValue(tvalues, target_idx_origin, frame_size_origin):
    _, frame_size, _ = tvalues.shape
    rem = int((frame_size_origin - frame_size)/2)
    target_idx = target_idx_origin - rem
    return tvalues[:, target_idx[1], target_idx[0]]

def cosSimular(vec1, vec2):
    norm1 = np.sqrt(np.sum(vec1**2))
    norm2 = np.sqrt(np.sum(vec2**2))
    cossim = np.sum(vec1*vec2)/norm1/norm2
    return cossim
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdata_dir', type = str, required = True,
                        help = '/path/to/row_data, contains yyyy-mm-dd : satellite')
    parser.add_argument('--gdata_dir', type = str, required = True,
                        help = '/path/to/CSV, contains yyyymmdd_1sec.csv : ground')
    parser.add_argument('--date', type = str, nargs = '+', required = True,
                        help = 'objective date, must be [yyyy-mm-dd]')
    parser.add_argument('--region_name', type = str, required = True,
                        help = 'region name, under /row_data/yyyy-mm-dd/pickles, like Tokyo')
    parser.add_argument('--limit_frame', type = int, default = 999,
                        help = 'number of frames for utilize, default [999](all)')
    parser.add_argument('--max_evals', type = int, default = 10,
                        help = 'max evaluation in hyper-parameter optimize [10]')
    args = parser.parse_args()

    pkldirs = [args.sdata_dir + '/' + d + '/pickles/' + args.region_name\
               for d in args.date]
    attrs = [(pkldir, args.gdata_dir, d, args.region_name,
              args.limit_frame, args.max_evals)
             for pkldir, d in zip(pkldirs, args.date)]

    result = OrderedDict()
    num_cores = int(input('input utilize core number : '))
    if num_cores <= 1:
        result['result'] = [_wrapper(attr) for attr in attrs]
    else:
        pool = Pool(num_cores)
        result['result'] = list(pool.map(_wrapper, attrs))
        pool.close()
        
    result['config'] = args

    with open('./result.pkl', 'wb') as f:
        pickle.dump(result, f)
            
if __name__ == '__main__':
    main()
