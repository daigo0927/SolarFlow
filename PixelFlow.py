# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

# take pre/post frame
# return pixel flow: shape(ygrid, xgrid, [flow_x,flow_y])
def PixelFlow(preframe,
              postframe,
              SearchRange = 5, # range for search pixel (radius)
              NeighborRange = 2): # range for treat as neightbor (radius)

    f_size = preframe.shape[0]

    s_range = SearchRange
    n_range = NeighborRange
    
    pixelflow = np.array([[PixelSearch(\
                            SearchArea = postframe[y-s_range:y+s_range+1,
                                                   x-s_range:x+s_range+1],
                            pixels = preframe[y-n_range:y+n_range+1,
                                              x-n_range:x+n_range+1])\
                           - np.array([s_range, s_range]) \
                           for x in np.arange(s_range, f_size-s_range)]\
                          for y in np.arange(s_range, f_size-s_range)])

    return pixelflow
    


def PixelSearch(SearchArea, # area for search flow
                pixels): # objective pixel, and surrounding pixels

    S_size = SearchArea.shape
    p_size = np.array(pixels.shape)
    
    loss = np.array([[PixelLoss(PostArea = SearchArea[y:y+p_size[0],
                                                      x:x+p_size[1]],
                                PreArea = pixels,
                                losstype = 'MSE')\
                      for x in range(S_size[1] - p_size[1] + 1)]\
                     for y in range(S_size[0] - p_size[0] + 1)])
    
    min_idx = np.array([i[0] for i in np.where(loss == np.min(loss))])
    # return center position of the pixels
    min_idx = min_idx[::-1] + (p_size-1.)/2

    # return pixel movement ([xflow, yflow])
    return min_idx

        

# compare 2 areas
def PixelLoss(PostArea, PreArea, losstype = 'MSE'):

    if losstype == 'MSE':
        loss = np.sqrt((PostArea - PreArea)**2)
    elif loss == 'other':
        # other loss function
        pass

    return np.mean(loss)


# visualize pixel flow
# pixelflow shape(y_len, x_len, 2)
# pixelflow contents : [xflow, yflow]
# !!! sort axis as sns.heatmap, top-left:(0,0)
def visFlow(pixelflow, vismargin=5):

    y_len, x_len, _ = pixelflow.shape
    Y, X = np.mgrid[0:y_len, 0:x_len]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.quiver(X, Y, pixelflow[:,:,0], -pixelflow[:,:,1],
              facecolor = 'blue')
    ax.set_xlim([-vismargin, x_len+vismargin])
    ax.set_ylim([y_len+vismargin, -vismargin])


    
    

    
    

    


    
    
    
