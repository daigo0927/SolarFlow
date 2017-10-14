# coding:utf-8

import os, sys
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
import pickle
import argparse
import pdb

from tqdm import tqdm

from SolarFlow import SolarFlow, easySolarFlow
from SmoothFlow import SmoothInterp
from misc.visualize import draw_cloud
from misc.utils import LinearInterp

from multiprocessing import Pool


def _wrapper(attr):
    _process(*attr)

def _process(pkldir, date, region_name, limit_frame):

    print('processing data in {} start'.format(pkldir))

    # oroginal shade ratio
    with open(pkldir + '/shade.pkl', 'rb') as f:
        data = pickle.load(f)
    f_origin, w_origin, _ = data.shape

    # train data, fall every every two slices
    data_train = data[:limit_frame:2]

    # experiment setting
    limit_frame = int(limit_frame)
    s_range = int(7)
    n_range = int(2)
    fine = int(2)

    result = {}

    # linear interp : shape(217, 26, 26)

    return
    

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
    args = parser.parse_args()

    pkldirs = [args.data_dir + '/' + d + '/pickles/' + args.region_name\
               for d in args.date]
    attrs = [(pkldir, args.date, args.region_name,
              args.limit_frame) for pkldir in pkldirs]
    
    num_cores = int(input('input utilize core number : '))
    pool = Pool(num_cores)
    results = list(pool.map(_wrapper, attrs))
    pool.close()

    with open('./result.pkl', 'wb') as f:
        pickle.dump(results, f)
            
if __name__ == '__main__':
    main()


    
def compare_Himawari_ground(sec_stride = 150):

    # days which has download ground measured solar radiation data
    days = ['03-25', '03-29', '04-01', '04-05']
    days_ = [''.join(d.split('-')) for d in days]

    for d, d_ in zip(days, days_):
        # h:Himawari, g:ground (120th building)
        hpath = '/Users/Daigo/Data/ShadeRatio/TotalRatioSrc/row_data/2017-{}/pickles/Waseda'.format(d)
        hdata = load_pickles(hpath)
        gpath = '~/Data/ShadeRatio/120BuildingData/CSV/2017{}_1sec.csv'.format(d_)
        gdata = pd.read_csv(gpath, header = None)
        gdata = pad_gdata(np.array(gdata))

        x = np.linspace(9, 18, 217)
        x_ = np.linspace(9, 18, 32400/sec_stride+1)
        plt.title('2017 {}'.format(d), fontsize = 15)
        plt.xlabel('time', fontsize = 15)
        plt.ylabel('solar radiation $[W/m^2]$', fontsize = 15)
        plt.ylim(0, 1200)
        plt.plot(x, hdata['crop'][:, 19, 20], label = 'Himawari (total)')
        plt.plot(x, hdata['outer'][:, 19, 20], label = 'Himawari (outer)')
        plt.plot(x, gdata[32399:64800:sec_stride, 1], label = '120th Building')
        plt.legend(fontsize = 14)
        plt.savefig('./result/{}compare.png'.format(d))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sec_stride', type = int, default = 150,
                        help = 'second stride for plot each slices of ground data')
    args = parser.parse_args()

    compare_Himawari_ground(**vars(args))
