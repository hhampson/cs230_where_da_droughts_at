"""
Authors: Bennett Bolen, Hannah Hampson, Mo Sodwatana
February 24, 2021

This script is for building the dataset.
"""


import numpy as np
from data import precipitation, drought_index, temperature, soil_moisture
from tempfile import TemporaryFile
import pandas as pd


# Extract data from drought index script
DI_VALUES = drought_index.SIX_MONTH_VALUES
DI_YEARS = drought_index.YEARS
DI_LAT = drought_index.LAT
DI_LON = drought_index.LON

# Specify number of input variables
NUM_VARS = 4  # precipitation, max temp, min temp, and soil moisture


def main():
    # Extract data from precipitation script
    precipitation_values_large = precipitation.SIX_MONTH_VALUES
    precipitation_years = precipitation.YEARS
    precipitation_lat = precipitation.LAT
    precipitation_lon = precipitation.LON

    # Extract data from temperature script, does not need to be downsampled
    min_temp = temperature.SIX_MONTH_VALUES_TMIN
    max_temp = temperature.SIX_MONTH_VALUES_TMAX
    temperature_years = temperature.YEARS
    temperature_lat = temperature.LAT
    temperature_lon = temperature.LON

    # Extract data from soil moisture script
    sm_values_large = soil_moisture.SIX_MONTH_VALUES
    sm_years = [2015, 2016, 2017]
    sm_lat = soil_moisture.LAT
    sm_lon = soil_moisture.LON
    
    # Downsample input variables to fit drought index resolution
    precip = downsample(precipitation_lat, precipitation_lon, precipitation_values_large, precipitation_years, 'sum')
    sm = downsample(sm_lat, sm_lon, sm_values_large, sm_years, 'avg')
    
    y_nans = build_y(DI_VALUES)
    x_nans = build_x(precip,max_temp,min_temp,sm)

    # Drop any examples that have NaN values
    x, y = drop_nans(x_nans, y_nans)
    return y, x


def downsample(var_lat, var_lon, var_values_large, var_years, method):
    """
    Downsamples data spatially to align with drought index scale.

    Inputs:
        Method: either 'sum' or 'avg' the values being downsampled
    """
    grid_size = DI_LAT[0] - DI_LAT[1]
    values_array = np.zeros((len(var_years), len(DI_LAT), len(DI_LON)))
    for year_idx in range(len(var_years)):
        for lat_idx in range(len(DI_LAT)):
            for lon_idx in range(len(DI_LON)):
                _, fine_lat_idx = get_area_coords(var_lat, [(DI_LAT[lat_idx] - 0.5 * grid_size), (DI_LAT[lat_idx] + 0.5 * grid_size)])
                _, fine_lon_idx = get_area_coords(var_lon, [(DI_LON[lon_idx] - 0.5 * grid_size), (DI_LON[lon_idx] + 0.5 * grid_size)])
                if method == 'sum':
                    value = var_values_large[fine_lat_idx[1]:fine_lat_idx[0], fine_lon_idx[0]:fine_lon_idx[1]].sum()
                if method == 'avg':
                    value = np.mean(var_values_large[fine_lat_idx[1]:fine_lat_idx[0], fine_lon_idx[0]:fine_lon_idx[1]], axis=None)
                values_array[year_idx, lat_idx, lon_idx] = float(value)
    return values_array


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


def build_y(di_values):
    """
    Build y array from drought index array to input into model.
    Each lat, lon, during one year, corresponds to one training example.
    """
    m = di_values.shape[0] * di_values.shape[1] * di_values.shape[2] # number of training examples
    y = {}
    for year_idx in range(0,di_values.shape[0]):
        new_vals = np.reshape(di_values[year_idx, :, :], (1, di_values.shape[1] * di_values.shape[2]))
        y[year_idx] = new_vals
    y = np.concatenate([y[year_idx] for year_idx in range(0, di_values.shape[0])], axis=1)
    assert y.shape == (1, m)
    return y


def build_x(precip, max_temp, min_temp, sm):
    m = DI_VALUES.shape[0] * DI_VALUES.shape[1] * DI_VALUES.shape[2]  # number of training examples
    x = {}
    grid_area = DI_VALUES.shape[1] * DI_VALUES.shape[2]
    for year_idx in range(0,DI_VALUES.shape[0]):
        precip_row = np.reshape(precip[year_idx, :, :], (1, grid_area))
        max_temp_row = np.reshape(max_temp[year_idx, :, :], (1, grid_area))
        min_temp_row = np.reshape(min_temp[year_idx, :, :], (1, grid_area))
        sm_row = np.reshape(sm[year_idx, :, :], (1, grid_area))
        new_vals = [precip_row, max_temp_row, min_temp_row, sm_row]
        x[year_idx] = new_vals
    x = np.concatenate([x[year_idx] for year_idx in range(0, DI_VALUES.shape[0])], axis=2)
    x = np.reshape(x, (NUM_VARS, m))
    assert x.shape == (NUM_VARS, m)
    return x


def drop_nans(x_nans, y_nans):
    """
    Take in X and Y matrices and throws out training examples with NaN values in them.
    Does so by creating a pandas dataframe from X and Y matrices then using
    dropna().
    """
    # build pandas dataframe
    df_nans = pd.DataFrame(np.concatenate((x_nans, y_nans)))
    df_no_nans = df_nans.dropna(axis=1)  # drop examples with any values = nan
    array_no_nans = np.array(df_no_nans)  # convert to numpy array
    x_no_nans = array_no_nans[0:4, :]
    y_no_nans = array_no_nans[-1, :].reshape(1, -1)

    assert x_no_nans.shape[0] == NUM_VARS
    assert x_no_nans.shape[1] == y_no_nans.shape[1]  # = m
    m = x_no_nans.shape[1]
    print("Number of training examples: " + str(m))
    return x_no_nans, y_no_nans


Y, X = main()
Y = np.array(Y)
outfile = TemporaryFile()
# np.save('Y',Y)  # first test
# np.save('X',X)  # first test
np.save('Y_v2', Y)  # version 2
np.save('X_v2', X)  # version 2
