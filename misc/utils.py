# coding:utf-8
import numpy as np
import pdb


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
