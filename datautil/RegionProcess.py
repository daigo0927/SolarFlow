import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pdb
import time
from tqdm import tqdm
import argparse
from glob import glob
from multiprocessing import Pool

from OuterRadiation import OuterRadiation
from CropTotalRadiation import CropTotalRadiation

def wrapper(attr):
    process(*attr)

def process(data_dir, d, longitude, latitude,
            t_range, num_frames, num_frames_fine, region):
    
    datepath = data_dir + '/' + d
    Rdatapath = datepath + '/Rdata_jp/'
    print('processing Rdata in {}'.format(Rdatapath))
    
    d_split = d.split('-')
    d_num = [int(d_s) for d_s in d_split]
    d_start = np.array([d_num[0], d_num[1], d_num[2], t_range[0], 0])
    d_end = np.array([d_num[0], d_num[1], d_num[2], t_range[1], 0])
    print('time range from {} to {}'.format(d_start, d_end))
    
    outerrad1 = OuterRadiation(latitude = latitude,
                               longitude = longitude,
                               date = d_start)
    outer = outerrad1.compute(number = num_frames, interval = 2.5)
    outerrad2 = OuterRadiation(latitude = latitude,
                               longitude = longitude,
                               date = d_start)
    outer_fine = outerrad2.compute(number = num_frames_fine, interval = 1/6)
    print('outer shape {}, outer_fine shape {}'.format(outer.shape, outer_fine.shape))

    print('processing total radiation ...')
    croptotal = CropTotalRadiation(latitude = latitude,
                                   longitude = longitude,
                                   start_date = d_start,
                                   end_date = d_end,
                                   data_path = Rdatapath)
    crop = croptotal.Crop()
    print('crop shape {}'.format(crop.shape))
    
    shade = 1. - crop/np.array(outer)
    print('shade shape {}'.format(shade.shape))

    pickles_path = datepath + '/pickles'
    if not os.path.isdir(pickles_path):
        os.mkdir(pickles_path)
    pickles_region_path = pickles_path + '/' + region
    if not os.path.isdir(pickles_region_path):
        os.mkdir(pickles_region_path)

    print('save pickle data ...')
    with open(pickles_region_path + '/outer.pkl', 'wb') as f:
        pickle.dump(outer, f)
    with open(pickles_region_path + '/outer_fine.pkl', 'wb') as f:
        pickle.dump(outer_fine, f)
    with open(pickles_region_path + '/crop.pkl', 'wb') as f:
        pickle.dump(crop, f)
    with open(pickles_region_path + '/shade.pkl', 'wb') as f:
        pickle.dump(shade, f)

    print('{} data process complete'.format(d))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--longitude_range', type = float, nargs = 2, required = True,
                        help = 'objecrive longitude range, must be [xxx.xx5, xxx.xx5]')
    parser.add_argument('--latitude_range', type = float, nargs = 2, required = True,
                        help = 'objecrive latitude range, must be [xx.xx5, xx.xx5]')
    parser.add_argument('--data_dir', type = str, required = True,
                        help = '/path/to/row_data, contains [yyyy-mm-dd]')
    parser.add_argument('--date', type = str, nargs = '+', required = True,
                        help = 'objective date, must be [yyyy-mm-dd]')
    parser.add_argument('--time_range', type = int, nargs = 2, default = [9, 18],
                        help = 'processing time range, default [9, 18]')
    parser.add_argument('--region_name', type = str, required = True,
                        help = 'region name, like Tokyo')
    args = parser.parse_args()
    
    lon_range = np.array(args.longitude_range)
    longitude = np.linspace(lon_range[0], lon_range[1],
                            round((lon_range[1] - lon_range[0])/0.01+1))
    lat_range = np.array(args.latitude_range)
    latitude = np.linspace(lat_range[0], lat_range[1],
                           round((lat_range[1] - lat_range[0])/0.01+1))
    print('confirm longitude, ', len(longitude), longitude)
    print('confirm latitude, ', len(latitude), latitude)

    t_range = np.array(args.time_range)
    num_frames = int((t_range[1] - t_range[0])*60/2.5 + 1) # each 2.5 minutes
    num_frames_fine = int((t_range[1] - t_range[0])*60*60/10 + 1) # each 10 seconds

    region = args.region_name

    attrs = [(args.data_dir, d, longitude, latitude,
              t_range, num_frames, num_frames_fine, region)\
             for d in args.date]
    num_cores = int(input('input core number : '))
    pool = Pool(num_cores)
    pool.map(wrapper, attrs)
    pool.close()

    
if __name__ == '__main__':
    main()
