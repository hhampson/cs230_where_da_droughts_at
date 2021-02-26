"""
Authors: Bennett Bolen, Hannah Hampson, Mo Sodwatana
February 24, 2021
"""

# TODO: downscale temporally extracting correct 6 month periods corresponding to our years

import netCDF4 as nc
import numpy as np
from numpy import newaxis
from variables import *  # imports variables such as lat_lims, lon_lims
from datetime import date
from dateutil.relativedelta import relativedelta


def main():
    # Reference SPEI netcdf file
    filename = '/Users/hannahhampson/OneDrive_Stanford_Old/OneDrive - Stanford/CS_230/project/pycharm/data/spei06.nc'
    # view_data(filename)
    total_values, lat, lon, time, values = extract_data(filename)
    six_month_values, years = extract_times(values, time)
    return six_month_values, years, lat, lon


def view_data(filename):
    """
    Opens up netcdf file to view its content, such as variables and dimensions.
    """
    data = nc.Dataset(filename)
    print(data)
    print(data.variables['time'])  # example of looking at 1 variable (time)
    print(data.variables['spei'])


def extract_data(filename):
    data = nc.Dataset(filename)
    values = data.variables['spei'][:]  # SPEI
    values[values > 2] = None
    time_days = data.variables['time'][:]  # days since 1901-01-01
    time = convert_time(time_days)
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
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


def convert_time(time_days):
    """
    Time is in days since 1900-01-01 00:00:0.0 so we need to convert to
    datetimes.
    """
    start_time = date(1900, 1, 1)
    delta_days = relativedelta(days=1)
    datetimes = start_time + time_days * delta_days
    return datetimes


def extract_times(values, time):
    start_idx = find_start_idx(time)  # start at the year we care about, year_start
    six_month_values = np.empty([1, values.shape[1], values.shape[2]])
    years = []
    for i in range(start_idx, len(time)):
        if time[i].month == dry_month_end:  # find six month SPEI corresponding to dry 6 month period
            years.append(time[i].year)
            period_values = values[i, :, :]
            period_values = period_values[newaxis, :, :]
            six_month_values = np.concatenate((six_month_values, period_values))
    six_month_values = six_month_values[1:, :, :]  # cut off empty first layer
    return six_month_values, years


def find_start_idx(time):
    for i in range(len(time)):
        if time[i].year == year_start:
            return i


if __name__ == '__main__':
    main()
