# coding: utf-8

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as anima
import seaborn as sns

from PixelFlow import PixelFlow, visFlow

with open('/Users/Daigo/Data/ShadeRatio/Machida/2016_8_20_9-12/pickles/ShadeRatio.pkl',
          'rb') as f:
    data = pickle.load(f)

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

def DoubleVis(cloud_origin, cloud_interp, interval = 200):

    n_origin, _, w_origin = cloud_origin.shape
    n_interp, _, w_interp = cloud_interp.shape

    n_rep = int((n_interp - 1)/(n_origin - 1))

    fig = plt.figure(figsize = (11, 4))

    def update(i, clouds1, clouds2):
        if i != 0:
            plt.cla()

        cloud1 = clouds1[i//n_rep]
        cloud2 = clouds2[i]

        c_size, _ = cloud2.shape
        drop = int((cloud1.shape[0] - c_size)/2)
            
        plt.subplot(121)
        sns.heatmap(cloud1[drop:drop+c_size, drop:drop+c_size],
                    vmin = 0, vmax = 1,
                    cmap = 'YlGnBu_r', annot = False)
        plt.title('original radiation')

        plt.subplot(122)
        sns.heatmap(cloud2,
                    vmin = 0, vmax = 1,
                    cmap = 'YlGnBu_r', annot = False)
        plt.title('interpolated radiation')

    ani = anima.FuncAnimation(fig, update,
                              fargs = (cloud_origin, cloud_interp),
                              interval = interval,
                              frames = n_interp)

    return ani

    
    

    
    

    
    
if __name__ == '__main__':
    for i in range(10):
        draw_flow(preframe = data[i], postframe = data[i+1])
        plt.savefig('/Users/Daigo/Desktop/tmp/flow{}.png'.format(i))
        plt.close()


    


