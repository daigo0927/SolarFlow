import os, sys
sys.path.append(os.pardir)
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from datautil.io import *

def load_double_data(day, sec_stride = 150): # day:'mm-dd'
    # return satellite(Himawari) and ground measured data
    day_ = ''.join(day.split('-')) # day_:'mmdd'
    
    hpath = '/Users/Daigo/Data/ShadeRatio/TotalRatioSrc/row_data/2017-{}/pickles/Waseda'.format(day)
    hdata = load_pickles(hpath)
    gpath = '~/Data/ShadeRatio/120BuildingData/CSV/2017{}_1sec.csv'.format(day_)
    processer = gdata_preprocesser(gpath)
    gdata = processer.move_avg(step = sec_stride)

    return hdata, gdata
    

def compare_Himawari_ground(sec_stride = 150):

    # days which has download ground measured solar radiation data
    days = ['03-25', '03-29', '04-01', '04-05']
    
    for d in days:
        hdata, gdata = load_double_data(d, sec_stride)
        # h:Himawari, g:ground (120th building)
        # hpath = '/Users/Daigo/Data/ShadeRatio/TotalRatioSrc/row_data/2017-{}/pickles/Waseda'.format(d)
        # hdata = load_pickles(hpath)
        # gpath = '~/Data/ShadeRatio/120BuildingData/CSV/2017{}_1sec.csv'.format(d_)
        # processer = gdata_preprocesser(gpath)
        # gdata = processer.move_avg(step = sec_stride)
        # gdata_fine = processser.move_avg(10)
        # gdata = pd.read_csv(gpath, header = None)
        # gdata = pad_gdata(np.array(gdata))

        x = np.linspace(9, 18, 217)
        x_ = np.linspace(9, 18, 32400/sec_stride+1)
        plt.title('2017 {}'.format(d), fontsize = 15)
        plt.xlabel('time', fontsize = 15)
        plt.ylabel('solar radiation $[W/m^2]$', fontsize = 15)
        plt.ylim(0, 1200)
        plt.plot(x, hdata['crop'][:, 19, 20], label = 'Himawari (total)')
        plt.plot(x, hdata['outer'][:, 19, 20], label = 'Himawari (outer)')
        plt.plot(x, gdata, label = '120th Building')
        plt.legend(fontsize = 14)
        plt.savefig('./result/{}compare.pdf'.format(d))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sec_stride', type = int, default = 150,
                        help = 'second stride for plot each slices of ground data')
    args = parser.parse_args()

    compare_Himawari_ground(**vars(args))
