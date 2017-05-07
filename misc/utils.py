# coding:utf-8
import numpy as np
import pdb

from scipy.interpolate import interp1d

def Time2Str(year = 2016,
             month = 1,
             day = 1,
             hour = 12,
             minute = 0,
             second = 0,
             sep = '-'):

    year_str = str(year)

    if month < 10:
        month_str = '0' + str(month)
    else:
        month_str = str(month)

    if day < 10:
        day_str = '0' + str(day)
    else:
        day_str = str(day)

    if hour < 10:
        h_str = '0' + str(hour)
    else:
        h_str = str(hour)

    if minute < 10:
        m_str = '0' + str(minute)
    else:
        m_str = str(minute)
        
    if second < 10:
        s_str = '0' + str(second)
    else:
        s_str = str(second)

    timestr = sep.join([year_str, month_str, day_str,
                        h_str, m_str, s_str])

    return timestr

def Time2Strings(year = 2016,
                 month = 8,
                 day = 20,
                 start_hour = 9,
                 start_minute = 0,
                 start_second = 0,
                 end_hour = 12,
                 end_minute = 0,
                 end_second = 0,
                 grid_second = 10,
                 sep = '-'):

    h = start_hour
    m = start_minute
    s = start_second

    TimeStrList = []
    status = True

    while status:
        timestring = Time2Str(year = year,
                              month = month,
                              day = day,
                              hour = h,
                              minute = m,
                              second = s,
                              sep = sep)
        
        TimeStrList.append(timestring)
        
        s = s + grid_second
        m = m + s//60
        s = s%60

        h = h + m//60
        m = m%60

        # pdb.set_trace()

        if(h == end_hour and m == end_minute and s > end_second):
            status = False

    return TimeStrList

class LinearInterp:

    def __init__(self,
                 data,
                 crop = 5):

        self.data = data
        self.f_len, self.y_len, self.x_len = data.shape
        self.xgrid = np.arange(self.x_len)
        self.ygrid = np.arange(self.y_len)

        self.axis_origin = np.arange(self.f_len)
        self.axis_new = None

        self.interfuncs = None

        self.result = None

    def interp(self, fineness = 15):

        self.axis_new = np.linspace(0, self.f_len-1, (self.f_len-1)*fineness+1)

        self.interfuncs = [[interp1d(self.axis_origin, self.data[:, y, x])\
                            for x in self.xgrid]\
                           for y in self.ygrid]

        result = np.array([[self.interfuncs[y][x](self.axis_new)\
                            for x in self.xgrid]\
                           for y in self.ygrid])

        self.result = np.transpose(result, (2,0,1))

        
    

                
