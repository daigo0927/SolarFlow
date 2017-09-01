import numpy as np
import time
import datetime
import pandas as pd
import pdb
import rpy2.robjects as robjects
import sys,os
sys.path.append('/Users/Daigo/Desktop/row_data')

class CropTotalRadiation:

    def __init__(self,
                 # ex) around Tokyo
                 latitude = np.array([35.695, 35.705, 35.715]), 
                 longitude = np.array([139.725, 139.735, 139.745]), 
                 data_path = '/Users/Daigo/Desktop/row_data/2016-08-20/RData_jp/',
                 # [year, month, day, hoor, second])
                 start_date = np.array([2016, 8, 20, 10, 0]),
                 end_date = np.array([2016, 8, 20, 12, 0])
                 ):
        LatStart = 47.595
        LonStart = 119.995

        rowindex = -(latitude - LatStart)/0.01  #(1190., 1189., 1188.)
        colindex = (longitude - LonStart)/0.01  #(1973., 1974., 1975.)

        # crop target area
        self.rowindex = np.array(rowindex[::-1]+0.1, dtype = int)
        # (1188, 1189, 1190)
        self.colindex = np.array(colindex+0.1, dtype = int)
        # (1973, 1974, 1975)

        # crop result array
        self.Result = np.empty((0, len(self.rowindex), len(self.colindex)), float)
        
        # start, and end date
        self.sy, self.sm, self.sd, self.sh, self.ss = start_date
        self.ey, self.em, self.ed, self.eh, self.es = end_date

        # .RData path list
        self.data_path = data_path
        self.pathlist = []
        framerange = np.int((self.eh+self.es/60-self.sh-self.ss/60)/(2.5/60.)+ 1)
        
        for i in range(framerange):
            interval_sum = 2.5 * i # expect the data obtained for each 2.5 minutes

            if(self.sm < 10):
                mname = '0' + str(self.sm)
            else:
                mname = str(self.sm)

            if(self.sd < 10):
                dname = '0' + str(self.sd)
            else:
                dname = str(self.sd)

            h = np.int(self.sh + (self.ss + interval_sum)//60)
            if(h < 10):
                hname = '0' + str(h)
            else:
                hname = str(h)

            s = np.int((10 * (interval_sum//10))%60)
            if(s < 10):
                sname = '0' + str(s)
            else:
                sname = str(s)

            a = 1 + i%4
            aname = '0' + str(a)

            # pdb.set_trace()

            path = str(self.sy) + mname + dname + hname \
                   + sname + 'jp' + aname + '.RData'

            self.pathlist.append(path)

    def Crop(self):

        for path in self.pathlist:
            robjects.r['load'](self.data_path + path)

            mat = robjects.r['mat_data']
            mat = np.array(mat)
            mat_crop = mat[self.rowindex[0] : self.rowindex[0]+len(self.rowindex),
                           self.colindex[0] : self.colindex[0]+len(self.colindex)]
            self.Result = np.append(self.Result,
                                    mat_crop.reshape(1,
                                                     mat_crop.shape[0],
                                                     mat_crop.shape[1]),
                                    axis = 0)

        return self.Result


            
            

        
