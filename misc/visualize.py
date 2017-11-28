# coding: utf-8

import os,sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from PixelFlow import PixelFlow

def visflow(flow, margin = 5):

    y_len, x_len = flow.shape
    X, Y = np.meshgrid(range(x_len), range(y_len))

    fig = sns.plt.figure()
    ax = fig.add_subplot(111)
    ax.quiver(X, Y, flow[:,:,0], -flow[:,:,1],
              facecolor = 'blue')
    ax.set_xlim([-margin, x_len+margin])
    ax.set_ylim([y_len+margin, -margin])
    

def draw_flow(preframe, postframe):
    pre = preframe
    post = postframe

    flow = PixelFlow(preframe = pre, postframe = postframe)

    fig = plt.figure(figsize = (11, 4))
    ax1 = fig.add_subplot(121)
    sns.heatmap(pre,
                vmin = 0, vmax = 1,
                cmap = 'YlGnBu_r', annot = False)
    plt.title('shade ratio')

    ax2 = fig.add_subplot(122)
    x_len, y_len, _  = flow.shape
    X, Y = np.meshgrid(range(x_len), range(y_len))
    ax2.quiver(X, Y, flow[:, :, 0], -flow[:, :, 1],
               facecolor = 'blue')
    ax2.set_xlim(-5, x_len+5)
    ax2.set_ylim(y_len+5, -5)
    plt.title('pixel flow')


def draw_cloud(cloud1, cloud2):
    
    c_size, _ = cloud2.shape
    drop = int((cloud1.shape[0] - c_size)/2)

    fig = plt.figure(figsize = (11, 4))
    
    ax1 = fig.add_subplot(121)
    sns.heatmap(cloud1[drop:drop+c_size, drop:drop+c_size],
                vmin = 0, vmax = 1,
                cmap = 'YlGnBu_r', annot = False)
    plt.title('original radiation')

    ax2 = fig.add_subplot(122)
    sns.heatmap(cloud2,
                vmin = 0, vmax = 1,
                cmap = 'YlGnBu_r', annot = False)
    plt.title('interpolated radiation')

def draw_multicloud(clouds, method = ['origin', 'linear', 'for']):
    # clouds[0] must be original one

    num_cloud = len(method)

    c_size, _ = clouds[-1].shape
    drop = int((clouds[0].shape[0] - c_size)/2)

    if num_cloud <= 3:
        ncol = 3
        nrow = 1
    else:
        ncol = 2
        nrow = np.ceil(num_cloud/2)
        fig = plt.figure(figsize = (11, 4.5*nrow))

    fig = plt.figure(figsize = (5.5*ncol, 4*nrow))

    ax_origin = fig.add_subplot(nrow, ncol, 1)
    sns.heatmap(clouds[0][drop:drop+c_size, drop:drop+c_size],
                vmin = 0, vmax = 1,
                cmap = 'YlGnBu_r', annot = False)
    plt.title('original radiation')

    for i in np.arange(1,num_cloud):
        ax = fig.add_subplot(nrow, ncol, i+1)
        sns.heatmap(clouds[i],
                    vmin = 0, vmax = 1,
                    cmap = 'YlGnBu_r', annot = False)
        plt.title('method : {}'.format(method[i]))

def draw_fusioncloud(cloud_origin, cloud_fine, method = ''):

    c_size, _ = cloud_fine.shape
    drop = int((cloud_origin.shape[0] - c_size)/2)

    cloud_origin = cloud_origin[drop:drop+c_size,
                                drop:drop+c_size]

    for i in range(cloud_origin.shape[0]):
        cloud_origin[i, 20-i//3:] = cloud_fine[i, 20-i//3:]

    sns.heatmap(cloud_origin,
                vmin = 0, vmax = 1,
                cmap = 'YlGnBu_r', annot = False)
    plt.title('{} interp origin/interp'.format(method))

def visResult(result):
    
    cols = ['coral', 'grey', 'grey', 'royalblue',
            'yellow', 'lime', 'yellow', 'lime', 'yellow', 'lime']
    sns.set_pallete(cols)
    plt.ylim(0, 600)
    plt.ylabel('radiation error $[W/m^2]$', fontsize = 14)
    plt.title(result['error'][0][0])
    sns.boxplot(data = result['error'][0][1])
    # plt.savefig('./result.pdf')
    plt.show()


# flow field plausiblity
def vis_flowfield(flowfield, theta, check_dim, margin = 5, figname = 'tmp', delay = 1000):

    print('flow field checking ...')
    # 1:normal, 0:strange
    masks = _checkfield(flowfield, theta, check_dim)
    t_len, y_len, x_len = masks.shape

    X, Y = np.meshgrid(np.arange(x_len), np.arange(y_len))
    for i, (flow, mask) in enumerate(tqdm(zip(flowfield, masks))):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # normal vector
        ax.quiver(X, Y, flow[:,:,0]*mask, -(flow[:,:,1]*mask),
                  facecolor = 'blue', scale = 60)
        ax.set_xlim([-margin, x_len+margin])
        ax.set_ylim([y_len+margin, -margin])
        # strange vector
        ax.quiver(X, Y, flow[:,:,0]*(1-mask), -(flow[:,:,1]*(1-mask)),
                  facecolor = 'red', scale = 60)
        ax.set_xlim([-margin, x_len+margin])
        ax.set_ylim([y_len+margin, -margin])

        plt.savefig(figname + str(i).zfill(3) + '.png')
        
    print('Figure outputs completed! Making gif images.')
    os.system('convert {0}*.png -delay 1000 {0}.gif'.format(figname, delay))
    os.system('rm {}*.png'.format(figname))
    print('Gif image saved.')
    

def _checkfield(flowfield, theta, check_dim):

    t_len, y_len, x_len, _ = flowfield.shape
    
    assert check_dim in [0, (1, 2), (0, 1, 2)],\
    'choose checking dimension from (1,2):spatial, 0:temporal, (0, 1, 2):double'

    if check_dim == (1, 2):
        check_result = np.array([[[_check(vec = flowfield[t, y, x],
                                          vecs = flowfield[t,
                                                           max(y-1,0):y+2,
                                                           max(x-1,0):x+2],
                                          theta = theta)
                                   for x in np.arange(x_len)]\
                                  for y in np.arange(y_len)]\
                                 for t in np.arange(t_len)])
    elif check_dim == 0:
        check_result = np.array([[[_check(vec = flowfield[t, y, x],
                                          vecs = flowfield[max(t-1,0):t+2, y, x],
                                          theta = theta)
                                   for x in np.arange(x_len)]\
                                  for y in np.arange(y_len)]\
                                 for t in np.arange(t_len)])
    elif check_dim == (0, 1, 2):
        check_result1 = np.array([[[_check(vec = flowfield[t, y, x],
                                           vecs = flowfield[t,
                                                            max(y-1,0):y+2,
                                                            max(x-1,0):x+2],
                                           theta = theta)
                                    for x in np.arange(x_len)]\
                                   for y in np.arange(y_len)]\
                                  for t in np.arange(t_len)])
        check_result2 = np.array([[[_check(vec = flowfield[t, y, x],
                                           vecs = flowfield[max(t-1,0):t+2, y, x],
                                           theta = theta)
                                    for x in np.arange(x_len)]\
                                   for y in np.arange(y_len)]\
                                  for t in np.arange(t_len)])
        check_result = check_result1*check_result2

    return check_result
    
# target vector, and surrounding vectors
def _check(vec, vecs, theta):
    
    num_vecs = vecs.size/2
    reduction_axis = tuple(np.arange(vecs.ndim)[:-1])
    vec_mean = (np.sum(vecs, axis = reduction_axis) - vec)/(num_vecs - 1.)
    diff = np.sqrt(np.sum((vec_mean - vec)**2))

    return diff < theta # True:normal, False:strange
