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
from SmoothFlow import SmoothInterp
from misc.visualize import draw_cloud
from misc.utils import LinearInterp
from datautil.io import load_pickles, pad_gdata

from multiprocessing import Pool


def _wrapper(attr):
    return _process(*attr)

def _process(pkldir, gdata_dir, date, region_name, limit_frame):

    print('processing data in {} start'.format(pkldir))

    # oroginal shade ratio
    data = load_pickles(pkldir)
    # original total radiation data, shape(limit_frame, 40, 40)
    if limit_frame == -1:
        limit_frame = data['crop'].shape[0]
    tdata = data['crop'][:limit_frame]
    sdata = data['shade'][:limit_frame]
    f_origin, w_origin, _ = tdata.shape
    obj_idx = [19, 20] # ground measurement location
    # total radiation at objective location, shppe(217, )
    obj_total = tdata[:limit_frame, obj_idx[1], obj_idx[0]]
    # outer radiation at objective location, shape(3241, )
    f_fine = (limit_frame-1)*15+1
    obj_outer_fine = data['outer_fine'][:f_fine, obj_idx[1], obj_idx[0]]

    # load ground data
    gpath = gdata_dir + '/' + ''.join(date.split('-')) + '_1sec.csv'
    gdata_origin = pd.read_csv(gpath, header = None)
    gdata_origin = np.array(gdata_origin)
    gdata_origin = pad_gdata(gdata_origin)
    fin_gframe = 32400+(limit_frame-1)*150 # 18h frame
    gdata = gdata_origin[32399:fin_gframe:150, 1] # 9-18h, every 2.5min, shape(217,)
    gdata_fine = gdata_origin[32399:fin_gframe:10, 1] # 9-18h, every 10sec shape(3241,)

    # experiment setting
    limit_frame = int(limit_frame)
    s_range = int(5)
    n_range = int(2)
    fine = int(15)

    shades = OrderedDict()

    # linear interp : shape(3241, 30, 30)
    print('linear interpolating')
    lin = LinearInterp(data = sdata, crop = s_range)
    lin.interp(fineness = fine)
    shades['linear'] = lin.result

    # biflow interp : shape(3241, 30, 30)
    print('biflow interpolating')
    easyflow = easySolarFlow(data = sdata,
                             search_range = s_range,
                             neighbor_range = n_range)
    easyflow.interp(fineness = fine, method = 'bi')
    shades['bi'] = easyflow.result

    # proposed method 1, temporal smoothing : shape(217, 23?, 23?)
    print('temporal smoothed interpolating ...')
    t_smooth = SmoothInterp(data = sdata,
                            search_range = s_range,
                            neighbor_range = n_range)
    t_smooth.interp(frameset_size = 3, # <- temporal smoothing
                    space_smooth_value = 10, fineness = fine)
    shades['temp_smooth'] = t_smooth.result
    
    # proposed method 2, spatial smoothing, and double smoothing : shape(217, 23?, 23?)
    s_smooth_values = np.array([6., 3., 1.5])
    labels = ['weak', 'middle', 'strong']
    for s_value, label in zip(s_smooth_values, labels):
        s_smooth = SmoothInterp(data = sdata,
                                search_range = s_range,
                                neighbor_range = n_range)
        print('{} spatial smoothed interpolating ...'.format(label))
        s_smooth.interp(frameset_size = 2,
                        space_smooth_value = s_value, fineness = fine)
        shades['space_smooth_{}'.format(label)] = s_smooth.result
        
        print('{} double smoothed interpolating ...'.format(label))
        s_smooth.interp(frameset_size = 3,
                        space_smooth_value = s_value, fineness = fine)
        shades['double_smooth_{}'.format(label)] = s_smooth.result


    # confirm interpolation results
    error = np.zeros((f_fine, len(shades.keys())+1))
    error[:,:] = None
    colnames = []
    # original 2.5min total radiation comparing
    error[::15,0] = ((obj_total - gdata)**2)**(1/2)
    colnames.append('origin')
    for i, key in enumerate(shades.keys()):
        shade = shades[key]
        f, w, _ = shade.shape
        drop = int((w_origin - w)/2)
        obj_shade = shade[:, obj_idx[1]-drop, obj_idx[0]-drop]
        obj_total_fine = obj_outer_fine[:f]*(1. - obj_shade)
        error[:len(obj_total_fine), i+1] =\
                            ((obj_total_fine - gdata_fine[:f])**2)**(1/2)
        colnames.append(key)
        
    error = pd.DataFrame(error)
    error.columns = colnames
    print(error.describe())
        
    return [region_name+date, error]
    

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
    parser.add_argument('--limit_frame', type = int, default = -1,
                        help = 'number of frames for utilize, default [-1](all)')
    args = parser.parse_args()

    pkldirs = [args.sdata_dir + '/' + d + '/pickles/' + args.region_name\
               for d in args.date]
    attrs = [(pkldir, args.gdata_dir, d, args.region_name,
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
    
