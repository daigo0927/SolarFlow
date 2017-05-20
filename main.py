# coding:utf-8

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from tqdm import tqdm

from SolarFlow import SolarFlow
from misc.visualize import draw_cloud
from misc.utils import LinearInterp


args = ['main.py', './somesolar.pkl', '-1', '5', '2']
args[:len(sys.argv)] = sys.argv

def main():

    _, datapath, limit_frame, Srange, Nrange = args
    limit_frame = int(limit_frame)
    Srange = int(Srange)
    Nrange = int(Nrange)

    with open(datapath, 'rb') as f:
        data = pickle.load(f)

    if limit_frame > 0:
        print('interp only data[:{}]'.format(limit_frame))
        data = data[:int(limit_frame)]

    sflow = SolarFlow(data = data,
                      SearchRange = Srange,
                      NeighborRange = Nrange)

    method = input('select interpolation method [linear/for/bi] : ')
    fine = int(input('interpolation fineness : '))
    if method == 'linear':
        lin = LinearInterp(data = data, crop = Srange)
        lin.interp(fineness = fine)
        sflow.result = lin.result
    else:
        sflow.interp(fineness = fine, method = method)
    
    save = input('save interp-result image? [y/n] : ')
    if save == 'y':
        if not os.path.exists('./image'):
            os.mkdir('./image')
        for i in tqdm(range(sflow.result.shape[0])):
            draw_cloud(sflow.data[i//fine], sflow.result[i])
            plt.savefig('./image/{}{}.png'.format(method, i))
            plt.close()
    elif save == 'n':
        print('not save interp-result')
        pass

    print('successfully finished')
    
    return sflow.result
            
if __name__ == '__main__':
    main()
