# import packages
from scipy.io import netcdf
from netCDF4 import Dataset as NetCDFFile
import pyproj
import numpy as np
import ftplib
import os

from data.variables import *  # imports variables such as lat_lims, lon_lims


def view_data(filename):
    """
    Opens up netcdf file to view its content, such as variables and dimensions.
    """
    data = NetCDFFile(filename)
    print(data)
    print(data.variables['time'])


def FTPimprort(FILE_NAME):
    """
    Open up temperature netcdf file and extract content.
    File data is from https://psl.noaa.gov/thredds/catalog/Datasets/cpc_global_temp/catalog.html
    """
    path = '/Datasets/cpc_global_temp/'  # path is the location of the file in the ftp server
    FILE_NAME = 't%s.%d.nc' % (data_type, year)  # filename is the name + extension of the file

    # connect to FTP server and download file
    ftp = ftplib.FTP("ftp2.psl.noaa.gov")
    ftp.login()
    ftp.cwd(path)
    ftp.retrbinary("RETR " + FILE_NAME, open(FILE_NAME, 'wb').write)
    ftp.quit()

    nc_data = NetCDFFile(FILE_NAME)  # load NetCDF file
    os.remove(FILE_NAME)  # remove downloaded file

    return nc_data


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


def extract_temp_data(data, lat_lims_var, lon_lims_var, data_type):
    """
    Open up temperature netcdf file and extract content, including clipping to the
    latitude and longitude limits of interest (specified in variables.py).
    """
    values = data.variables['t' + data_type][:]  # tmin or tmax
    time = data.variables['time'][:]
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:] - 360
    area_lat, idx_lat = get_area_coords(lat, lat_lims_var)
    area_lon, idx_lon = get_area_coords(lon, lon_lims_var)
    area_values = values[:, idx_lat[0]:idx_lat[1], idx_lon[0]:idx_lon[1]]
    dict_temp = {'values': values,
                 'area_lat': area_lat,
                 'area_lon': area_lon,
                 'time': time,
                 'area_values': area_values}
    return dict_temp


def average_temp_data(area_value, from_year, to_year):
    """
    Takes daily temperature and averages total over the "wet"
    six months of the year (specified in variables.py).
    Returns:
        six_month_values: np.array with dimensions (year index, lat, lon)
    """
    area_average = np.empty((0, area_value.shape[1], area_value.shape[2]))
    for i in range(0, to_year - from_year, 1):
        area = np.mean(area_value[(305 + (365 * i)):(305 + (365 * i)) + 150, :, :], axis=0)
        area_average = np.concatenate((area_average, np.reshape(area, (1, area_value.shape[1], area_value.shape[2]))))
    return area_average

def monthly_average(temp, area_value, from_year, to_year):
    area_average = np.empty((0, area_value.shape[1], area_value.shape[2]))
    
    for month in range(0,12 * 4):
        area = np.mean(temp[30*month:30*month+30,:,:],axis = 0)
        area_average = np.concatenate((area_average, np.reshape(area, (1, area_value.shape[1], area_value.shape[2]))))
    
    return area_average

# convert start and end year variable input to using input
from_year = year_start
to_year = year_end + 1
types = ["max", "min"]

# convert lat and long variable input to using input
lat_lims_var = [lat_lims[0] - 1, lat_lims[1] -1]
lon_lims_var = lon_lims

dict_tmax = dict()
dict_tmin = dict()

# for loop over the years and max and min
for year in range(from_year, to_year + 1, 1):  # to include to_year
    for data_type in types:
        FILE_NAME = 't%s.%d.nc' % (data_type, year)  # filename + extension of the file
        # path = 'drive/Shareddrives/CS230 Project/preprocessing_temperature/' + FILE_NAME

        # load NetCDF max and min files
        if data_type == "max":
            data = FTPimprort(FILE_NAME)
            dict_tmax[year] = extract_temp_data(data, lat_lims_var, lon_lims_var, data_type)

        else:
            data = FTPimprort(FILE_NAME)
            dict_tmin[year] = extract_temp_data(data, lat_lims_var, lon_lims_var, data_type)

# combine data from all years
tmin_all = np.concatenate([dict_tmin[year]['area_values'] for year in range(from_year, to_year + 1)])
tmax_all = np.concatenate([dict_tmax[year]['area_values'] for year in range(from_year, to_year + 1)])

tmin_all[tmin_all < -9e+30] = None
tmax_all[tmax_all < -9e+30] = None

# create 6 month average values
SIX_MONTH_VALUES_TMIN = average_temp_data(tmin_all, from_year, to_year)
SIX_MONTH_VALUES_TMAX = average_temp_data(tmax_all, from_year, to_year)

LAT = dict_tmin[from_year]['area_lat']
LON = dict_tmin[from_year]['area_lon']
YEARS = np.arange(from_year, to_year)

# monthly averages values
monthly_area_average_min = monthly_average(tmin_all, tmin_all, from_year, to_year)
print(monthly_area_average_min[1][:][:])

monthly_area_average_max = monthly_average(tmax_all, tmax_all, from_year, to_year)
print(np.min(monthly_area_average_max))

np.save("MINTEMP_monthly.npy", np.array(monthly_area_average_min))
np.save("MAXTEMP_monthly.npy", np.array(monthly_area_average_max))
np.save("LAT_TEMP.npy", np.array(LAT))
np.save("LON_TEMP.npy", np.array(LON))
