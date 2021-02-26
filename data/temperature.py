# import packages
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.io import netcdf
from netCDF4 import Dataset as NetCDFFile 
import pyproj
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import ftplib
import os

from data.variables import *  # imports variables such as lat_lims, lon_lims

def view_data(filename):
  # Opens up the file to view its content, such as variables and dimensions.
  # Assumes netcdf file format.
    data = NetCDFFile(filename)
    print(data)
    print(data.variables['time'])

def FTPimprort(FILE_NAME):
  path = '/Datasets/cpc_global_temp/' # path is the location of the file in the ftp server
  FILE_NAME = 't%s.%d.nc' % (data_type, year) # filename is the name + extension of the file 

  # connect to FTP server and download file
  # https://psl.noaa.gov/thredds/catalog/Datasets/cpc_global_temp/catalog.html
  # ftp login is anonymous
  # ftp.cwd will change the current working directory to where the file is located in order to download it
  # retrbinary will get the file from the server and store in your local machine using the same name it had on the server
  ftp = ftplib.FTP("ftp2.psl.noaa.gov") 
  ftp.login() 
  ftp.cwd(path)
  ftp.retrbinary("RETR " + FILE_NAME ,open(FILE_NAME, 'wb').write)
  ftp.quit()

  # load NetCDF file
  nc_data = NetCDFFile(FILE_NAME)
  
  # remove downloaded file
  os.remove(FILE_NAME)

  return nc_data

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

def extract_temp_data(data, lat_lims, lon_lims, data_type):
  values = data.variables['t'+ data_type][:]  # tmin or tmax
  time = data.variables['time'][:]
  lat = data.variables['lat'][:]
  lon = data.variables['lon'][:]
  area_lat, idx_lat = get_area_coords(lat, lat_lims[::-1]) # reverse latitide limits
  area_lon, idx_lon = get_area_coords(lon, lon_lims)
  area_values = values[:, idx_lat[0]:idx_lat[1], idx_lon[0]:idx_lon[1]]
  dict_temp = {'values': values,
          'area_lat': area_lat,
          'area_lon': area_lon,
          'time': time,
          'area_values': area_values}
  return dict_temp

def average_temp_data(area_value):
  # Averages daily temperatures November to March.
  area_average = np.empty((0,area_value.shape[1], area_value.shape[2]))
  for i in range(1,6,1):
    area = np.mean(area_value[(305*i):(305*i)+150,:,:],axis=0)
    area_average = np.concatenate((area_average,np.reshape(area,(1,area_value.shape[1], area_value.shape[2]))))
  return area_average

# input range of years
from_year = 2015
to_year = 2019
types = ["max", "min"]

# input lat and lon over California
lat_lims_var = [lat_lim[1], lat_lim[0]] # latitude range over California
lon_lims_var = [lon_lim[0] + 360, lon_lim[1] + 360] # longitude range over California

dict_tmax = dict()
dict_tmin = dict()

# for loop over the years
for year in range(from_year, to_year+1, 1): # to include to_year
    for data_type in types:
        FILE_NAME = 't%s.%d.nc' % (data_type, year) # filename + extension of the file
        # path = 'drive/Shareddrives/CS230 Project/preprocessing_temperature/' + FILE_NAME

        # load NetCDF file
        if data_type == "max":
          data = FTPimprort(FILE_NAME)
          dict_tmax[year] = extract_temp_data(data, lat_lims_var, lon_lims_var, data_type)

        else:
          data = FTPimprort(FILE_NAME)
          dict_tmin[year] = extract_temp_data(data, lat_lims_var, lon_lims_var, data_type)

# combine data from all years
tmin_all = np.concatenate([dict_tmin[year]['area_values'] for year in range(from_year, to_year+1)])
tmax_all = np.concatenate([dict_tmax[year]['area_values'] for year in range(from_year, to_year+1)])

# create average values
tmin_avg = average_temp_data(tmin_all)
tmax_avg = average_temp_data(tmax_all)

LAT = dict_tmin[from_year]['area_lat']
LON = dict_tmin[from_year]['area_lon'] 
TIME = dict_tmin[from_year]['time'] 
SIX_MONTH_VALUES_TMIN = tmin_avg
SIX_MONTH_VALUES_TMAX = tmax_avg
