
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

from optimize import VectorOptimizer
from datautil.io import load_pickles

from multiprocessing import Pool

class OptimizeProcessor(VectorOptimizer):

    def __init__(self,
                 fullframes,
                 neighbor_range = 2, search_range = 5,
                 losstype = 'MSE'):
        super().__init__(fullframes,
                         neighbor_range, search_range,
                         losstype)
        self.vec_record = None
        self.lossrecord = None

    def optimize(self, num_iter, coef):
        self.coef = coef
        self.vec_record = []
        self.lossrecord = np.zeros((num_iter))
        print('optimize vector field, and record loss')
        for i in tqdm(range(num_iter)):
            reg_s, reg_t, reg_ts = reg_all(self.vec_field, self.vec_range)
            # concatenate matching-loss and regularizers
            self.loss_concat = self._concat(reg_s, reg_t, reg_ts)
            self.vec_field = np.array([[[get_vector(l, coef = coef)
                                         for l in loss]\
                                        for loss in loss_frame]\
                                       for loss_frame in self.loss_concat])
            # record regularized loss
            self.vec_record.append(self.vec_field)
            self.lossrecord[i] = np.sum(self.loss)

        return self.vec_record, self.lossrecord

    
def _wrapper(attr):
    return _process(*attr)

def _process(pkldir, date, region_name, limit_frame, num_iter, coef):

    print('processing data in {} start'.format(pkldir))
    data = load_pickles(pkldir)

    processor = OptimizeProcessor(fullframes = data['shade'][:limit_frame],
                                  neighbor_range = 2,
                                  search_range = 5,
                                  losstype = 'MSE')
    vec_record, lossrecord = processor.optimize(num_iter, np.array(coef))

    return {'vectors':vec_record, 'losses':lossrecord}


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
    parser.add_argument('--num_iter', type = int, default = 10,
                        help = 'optimize iteration [10]')
    parser.add_argument('--coef', type = int, nargs = 3, required = True,
                        help = 'three regularizer coefficients, like [0.01, 0.01, 0.005]')
    args = parser.parse_args()

    pkldirs = [args.data_dir + '/' + d + '/pickles/' + args.region_name\
               for d in args.date]
    attrs = [(pkldir, d, args.region_name,
              args.limit_frame, args.num_iter, args.coef)
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
