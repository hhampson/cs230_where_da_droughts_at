"""
Authors: Bennett Bolen, Hannah Hampson, Mo Sodwatana
February 24, 2021
"""

# TODO: fix convert_to_monthly - currently not concatenating correctly, and then avg over 6 month period

import netCDF4 as nc
from ftplib import FTP
import numpy as np
from numpy import newaxis
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from variables import *  # imports variables such as lat_lims, lon_lims


def main():
    filename = 'X128.12.122.45.54.19.46.11.nc'  # retrieved from the NOAA website from specifying subset boundaries
    # download_data(filename)
    # view_data(filename)
    total_values, lat, lon, time, values = extract_data(filename)
    month_values, month_times = convert_to_monthly(values, time)
    # print(time[0:31])
    print(np.sum(values[0:31, :, :].sum(axis=0) - month_values[0, :, :]))
    print(time[0])
    print(month_times[0])
    print(values.shape)
    print(month_values[0, :, :])
    # print(len(month_times))


def download_data(filename):
    """
    Open up NOAA ftp server and download precipitation data. Unnecessary
    step if file already downloaded. After this step the netcdf file should
    be in the current folder.
    """
    ftp = FTP('ftp2.psl.noaa.gov')
    ftp.login()
    ftp.cwd('/Public/www')
    file = open(filename, 'wb')
    ftp.retrbinary('RETR ' + filename, file.write)
    ftp.quit()


def view_data(filename):
    """
    Opens up netcdf file to view its content, such as variables and dimensions.
    """
    data = nc.Dataset(filename)
    print(data)
    print(data.variables['time'])  # example of looking at 1 variable (time)
    print(data.variables['precip'])


def extract_data(filename):
    data = nc.Dataset(filename)
    values = data.variables['precip'][:]  # precipitation values in mm
    time_hrs = data.variables['time'][:]  # days since 1901-01-01
    time = convert_time(time_hrs)
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    lon = lon - 360  # convert from all positive longitudes to negative
    area_lat, idx_lat = get_area_coords(lat, lat_lims[::-1])  # reverse latitude limits
    area_lon, idx_lon = get_area_coords(lon, lon_lims)
    area_values = values[:, idx_lat[0]:idx_lat[1], idx_lon[0]:idx_lon[1]]
    return values, area_lat, area_lon, time, area_values


def get_area_coords(coordinate_list, lims):
    idx = []
    for lim in lims:
        minimum = float("inf")
        for i in range(len(coordinate_list)):
            if abs(lim - coordinate_list[i]) < minimum:
                final_value = i
                minimum = abs(lim - coordinate_list[i])
        idx.append(final_value)
    area_coords = coordinate_list[idx[0]: idx[1]]
    return area_coords, idx


def convert_time(time_hrs):
    """
    Time is in hours since 1800-01-01 00:00:0.0 so we need to convert to
    datetimes.
    """
    start_time = date(1800, 1, 1)
    delta_hours = relativedelta(hours=1)
    datetimes = start_time + time_hrs * delta_hours
    return datetimes


def convert_to_monthly(values, time):
    """
    Converts daily precipitation data to monthly cumulative values.
    """
    month_idx = 0
    monthly_values = np.empty([1, values.shape[1], values.shape[2]])  # create empty array
    month_times = []
    for i in range(len(time)-1):
        if time[i].month != time[i+1].month:
            month_times.append(datetime(time[i].year, time[i].month, 1))  # add this month to new datetimes list
            month_values = values[month_idx:i, :, :].sum(axis=0)  # sum all the values from each day of that month
            month_values = month_values[newaxis, :, :]
            monthly_values = np.concatenate((monthly_values, month_values))
            month_idx = i + 1
    monthly_values = monthly_values[1:, :, :]
    return monthly_values, month_times


if __name__ == '__main__':
    main()