"""
Authors: Bennett Bolen, Hannah Hampson, Mo Sodwatana
February 24, 2021

This script accounts for the precipitation input data for our model. It processes daily gridded precipitation data
(courtesy of NOAA), and converts the data into a 4d array with dimensions (time, lat, lon). The data we are interested
in are cumulative precipitation over the wetter 6 month period in California, where that period is defined in
variables.py. This script goes through the file's time period, extracts each "wet" period, and sums daily precip (mm)
over each lat, lon.

The output is our 4d matrix for cumulative 6 month wet season precipitation, a list of corresponding years, a latitude
numpy array, and longitude numpy array.
"""


import os
import netCDF4 as nc
from ftplib import FTP
import numpy as np
from numpy import newaxis
from datetime import date
from dateutil.relativedelta import relativedelta
from variables import *  # imports variables such as lat_lims, lon_lims




def download_data(filename):
    """
    Open up NOAA ftp server and download precipitation data. Unnecessary
    step if file already downloaded. After this step the netcdf file should
    be in the location specified in NEW_FILENAME.
    """
    ftp = FTP('ftp2.psl.noaa.gov')
    ftp.login()
    ftp.cwd('/Public/www')
    file = open(filename, 'wb')
    ftp.retrbinary('RETR ' + filename, file.write)
    os.rename(filename, NEW_FILENAME)  # TODO: problem renaming because it takes a second to download
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
    """
    Open up precipitation netcdf file and extract content, including clipping to the
    latitude and longitude limits of interest (specified in variables.py).
    """
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


def convert_time(time_hrs):
    """
    Time is in hours since 1800-01-01 00:00:0.0 so we need to convert to
    datetimes.
    """
    start_time = date(1800, 1, 1)
    delta_hours = relativedelta(hours=1)
    datetimes = start_time + time_hrs * delta_hours
    return datetimes


def convert_to_six_month(values, time):
    """
    Takes daily precipitation and sums total over the "wet"
    six months of the year (specified in variables.py).

    Returns:
        six_month_values: np.array with dimensions (corresponding year, lat, lon)
        years: list of integers representing year associated with first dimension of six_month_values.
            Note that if wet year started November 2015 and ended April 2016, year = 2015.
    """
    six_month_values = np.empty([1, values.shape[1], values.shape[2]])
    start_idx = 0
    years = []
    for i in range(len(time)-1):
        if time[i].month == dry_month_end and time[i+1].month == wet_month_start:
            start_idx = i+1
            years.append(time[i].year)
        elif time[i].month == wet_month_end and time[i+1].month == dry_month_start:
            end_idx = i+1
            period_values = values[start_idx:end_idx, :, :].sum(axis=0)  # sum values from each day of 6 mo period
            period_values = period_values[newaxis, :, :]
            six_month_values = np.concatenate((six_month_values, period_values))
    six_month_values = six_month_values[1:, :, :]  # cut off empty first layer
    return six_month_values, years


def convert_to_monthly(values, time):
    # initialize monthly values matrix
    monthly_values = np.empty((0, values.shape[1], values.shape[2]))
    start_idx = 0
    months = []
    for i in range(len(time) - 1):
        if time[i].month != time[i-1].month:  # entering new month
            end_idx = i-1
            months.append(time[i-1].month)
            month_values = values[start_idx:end_idx, :, :].sum(axis=0)  # sum over previous month
            month_values = month_values[newaxis, :, :]
            monthly_values = np.concatenate((monthly_values, month_values))
            start_idx = i
        elif time[i].month == 1 and time[i].year == year_end + 1:  # end when we get past desired time period
            break
    assert monthly_values.shape[0] == 12 * (year_end - year_start + 1)
    return monthly_values


# v1 filename: X128.12.122.45.54.19.46.11.nc
# v2 filename:
FILENAME = 'X128.12.122.126.61.19.40.47.nc'  # name of file in NOAA server, w/ temporal and spatial boundaries specified
NEW_FILENAME = "~/data/precipitation.nc"  # store file to data folder in instance

# download_data(FILENAME)
# view_data(NEW_FILENAME)
# Variables of interest for build_dataset.py: six_month_values, years, lat, lon
TOTAL_VALUES, LAT, LON, TIME, VALUES = extract_data(FILENAME)
SIX_MONTH_VALUES, YEARS = convert_to_six_month(VALUES, TIME)
MONTHLY_VALUES = convert_to_monthly(VALUES, TIME)

np.save("../processed_data/PRECIP_monthly.npy", np.array(MONTHLY_VALUES))
np.save("../processed_data/LAT_PRECIP.npy", np.array(LAT))
np.save("../processed_data/LON_PRECIP.npy", np.array(LON))

