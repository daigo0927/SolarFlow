import numpy as np
import time
import datetime
import pandas as pd
import pdb
from tqdm import tqdm

class OuterRadiation:

    def __init__(self,
                 # ex) around Tokyo
                 latitude = np.array([35.695, 35.705, 35.715]), 
                 longitude = np.array([139.725, 139.735, 139.745]), 
                 date = np.array([2016,1,1,12,0])
                 # [year, month, day, hoor, second]
                 ):

        self.rownames = list(map(str, latitude[::-1]))
        self.colnames = list(map(str, longitude))
        self.dp = pd.Panel(major_axis = self.rownames, minor_axis = self.colnames)
        
        self.latitude = np.radians(latitude)
        self.longitude = np.radians(longitude)

        

        self.y, self.m, self.d, self.h, self.s = date

        days = datetime.datetime(self.y, self.m, self.d) - \
               datetime.datetime(self.y, 1, 1)
        self.dn = days.days + 1

        self.theta = 2 * np.pi * (self.dn - 1)/365

        # sun declination
        self.delta = 0.006918 - 0.399912 * np.cos(self.theta) \
                     + 0.070257 * np.sin(self.theta) \
                     - 0.006758 * np.cos(2 * self.theta) \
                     + 0.000907 * np.sin(2 * self.theta) \
                     - 0.002697 * np.cos(3 * self.theta) \
                     + 0.001480 * np.sin(3 * self.theta)

        # Geocentric solar distance
        self.r = 1 / np.sqrt(1.000110 + 0.034221 * np.cos(self.theta) \
                             + 0.001280 * np.sin(self.theta) \
                             + 0.000719 * np.cos(2 * self.theta) \
                             + 0.000077 * np.sin(2 * self.theta))

        # Uniform time difference
        
        self.Eq = - 0.0002786049 + 0.1227715 * np.cos(self.theta + 1.498311) \
                  - 0.1654575 * np.cos(2 * self.theta - 1.261546) \
                  - 0.0053538 * np.cos(3 * self.theta - 1.1571)

        self.StandardLatitude = np.radians(135.0)

        # tenmoral solar angle
        self.angle = None

        # sun orientation
        self.psi = None
        # sun altitude
        self.alpha = None

        # radiation out of air
        self.Q = None

    def compute(self, interval = 2.5, number = 10, save = False):

        yname = str(self.y)
        if(self.m < 10):
            mname = '0' + str(self.m)
        elif():
            mname = str(self.m)
        if(self.d < 10):
            dname = '0' + str(self.d)
        elif():
            dname = str(self.d)
            

        # for each time
        for i in tqdm(range(number)):

            df = pd.DataFrame(index = self.latitude[::-1])

            interval_sum = interval * i
            
            h = self.h + (self.s + interval_sum)//60
            if(h < 10):
                hname = '0' + str(h)
            elif():
                hname = str(h)

            s = (10 * interval_sum//10)%60
            if(s < 10):
                sname = '0' + str(s)
            elif():
                sname = str(s)

            a = 1 + i%4
            aname = '0' + str(a)
                
            # for each longtitude
            for lon in self.longitude:
                
                self.angle = (self.h + (self.s + interval * i)/60 - 12) \
                             * np.pi / 12 \
                             + (lon - self.StandardLatitude) \
                             + self.Eq * np.pi / 12

                Qseries = []
                # for each latitude
                for lat in self.latitude:
                    
                    self.alpha = np.arcsin(np.sin(lat) * np.sin(self.delta) \
                                           + np.cos(lat) \
                                           * np.cos(self.delta) \
                                           * np.cos(self.angle))
                    self.psi = np.arctan(np.cos(lat) * np.cos(self.delta) \
                                         * np.sin(self.angle) \
                                         / (np.sin(lat) * np.sin(self.alpha))\
                                         - np.sin(self.delta))
                    self.Q = 1367 * self.r**2 * np.sin(self.alpha)

                    Qseries.append(self.Q)
                    
                Qseries = Qseries[::-1]

                df[str(np.rad2deg(lon))] = Qseries

            index = [str(np.rad2deg(l)) for l in self.latitude[::-1]]
            df.index = index
            # pdb.set_trace()

            self.dp[i] = df

        return self.dp
