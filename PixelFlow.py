# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pdb

# take pre/post frame
# return pixel flow: shape(ygrid, xgrid, flow_x, flow_y)
def PixelFlow(preframe,
              postframe,
              crop = 7, # crop: discard outer circumference
              SearchRange = 7, # range for search pixel (radius)
              NeighborRange = 5): # range for treat as neightbor (radius)

    try:
        if crop < SearchRange:
            raise ValueError('search range exesses the given map size')
    except ValueError as e:
        print(e)
        
    def PixelSearch(SearchArea, # area for search flow
                    pixels): # objective pixel, and surrounding pixels

        S_size = SearchRange.shape
        p_size = pixels.shape

        loss = np.array([[PixelLoss(PostArea = SearchArea[y:y+p_size[0],
                                                          x:x+p_size[1]],
                                    PreArea = pixels,
                                    losstype = 'MSE')\
                          for x in range(S_size[1] - p_size[1] + 1)]\
                         for y in range(S_size[0] - p_size[0] + 1)])

        

# compare 2 areas
def PixelLoss(PostArea, PreArea, losstype = 'MSE'):

    if losstype == 'MSE':
        loss = np.sqrt((PostArea - PreArea)**2)
    elif loss == 'other':
        # other loss function
        pass

    return np.mean(loss)

    
    

    


    
    
    
