# coding: utf-8

import numpy as np
import pickle
import matplotlib.pyplot as plt
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

for i in range(10):
    draw_flow(preframe = data[i], postframe = data[i+1])
    plt.savefig('/Users/Daigo/Desktop/tmp/flow{}.png'.format(i))
    plt.close()


    


