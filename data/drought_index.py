"""
Authors: Bennett Bolen, Hannah Hampson, Mo Sodwatana
February 24, 2021

This script accounts for the drought index data as y values for our model. It processes Standardized Precipitation-
Evapotranspiration Index (SPEI) gridded data, and pulls out the time and area values of interest (over California), and
outputs a 4d array with dimensions (time, lat, lon). The data is already in a mean 6 month values, so this script pulls
out those values at the end of our "dry" periods that we are trying to predict with our model.

The output is our 4d matrix of average SPEI values for our 6 month dry seasons, a list of corresponding years, a
latitude numpy array, and longitude numpy array.
"""

import netCDF4 as nc
import numpy as np
from numpy import newaxis
from data.variables import *  # imports variables such as lat_lims, lon_lims
from datetime import date
from dateutil.relativedelta import relativedelta


def view_data(filename):
    """
    Opens up netcdf file to view its content, such as variables and dimensions.
    """
    data = nc.Dataset(filename)
    print(data)
    print(data.variables['time'])  # example of looking at 1 variable (time)
    print(data.variables['spei'])


def extract_data(filename):
    """
    Open up SPEI netcdf file and extract content, including clipping to the
    latitude and longitude limits of interest (specified in variables.py). Also
    converts fill values to "None".
    """
    data = nc.Dataset(filename)
    values = data.variables['spei'][:]  # SPEI
    values[values > 2] = None  # convert fill values (1e30) to None
    time_days = data.variables['time'][:]  # days since 1901-01-01
    time = convert_time(time_days)
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    area_lat, idx_lat = get_area_coords(lat, lat_lims[::-1])  # reverse latitude limits
    area_lon, idx_lon = get_area_coords(lon, lon_lims)
    area_values = values[:, idx_lat[0]:idx_lat[1], idx_lon[0]:idx_lon[1]]
    return values, area_lat, area_lon, time, area_values


def get_area_coords(coordinate_list, lims):
    """
    Given an array of coordinates and coordinate limits specified, return the
    index corresponding to those limits and the clipped array.
    """
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
    """
    Extract data for 6 month periods we're interested in.
    """
    start_idx = find_start_idx(time)  # start at the year we care about, year_start
    six_month_values = np.empty([1, values.shape[1], values.shape[2]])
    years = []
    for i in range(start_idx, len(time)):
        if time[i].year < (year_end + 2):  # so we don't exceed time frame we care about
            if time[i].month == dry_month_end:  # find six month SPEI corresponding to dry 6 month period
                years.append(time[i].year)
                period_values = values[i, :, :]
                period_values = period_values[newaxis, :, :]
                six_month_values = np.concatenate((six_month_values, period_values))
    six_month_values = six_month_values[1:, :, :]  # cut off empty first layer
    return six_month_values, years


def find_start_idx(time):
    """
    Find index of time corresponding to our start year (from variables.py).
    """
    for i in range(len(time)):
        if time[i].year == year_start + 1:  # add one because dry season here lags by 6 months, which is in next yr
            return i


# Variables of interest for build_dataset.py: six_month_values, years, lat, lon
# Reference SPEI netcdf file
FILENAME = '~/data/spei06.nc'
# view_data(filename)
TOTAL_VALUES, LAT, LON, TIME, VALUES = extract_data(FILENAME)
SIX_MONTH_VALUES, YEARS = extract_times(VALUES, TIME)
