# coding: utf-8

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as anima
import seaborn as sns

from PixelFlow import PixelFlow, visFlow


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

    


    
    
    



    


