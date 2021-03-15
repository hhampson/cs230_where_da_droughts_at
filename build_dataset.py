"""
Authors: Bennett Bolen, Hannah Hampson, Mo Sodwatana
February 24, 2021

This script is for building the dataset.
"""

import numpy as np
# from data import precipitation, drought_index, temperature, soil_moisture
from tempfile import TemporaryFile
import pandas as pd
from data.variables import *  # imports variables such as lat_lims, lon_lims


# Extract data from drought index script
DI_VALUES = np.load('processed_data/SIX_MONTH_VALUES_DI.npy')
DI_LAT = np.load('processed_data/LAT_DI.npy')
DI_LON = np.load('processed_data/LON_DI.npy')

# Specify number of input variables
NUM_VARS = 4  # precipitation, max temp, min temp, and soil moisture


def main():
    # If drought index matrix 4 years, then cut off last irrelevant year
    if DI_VALUES.shape[0] > 3:
        di = DI_VALUES[0:3, :, :]
    else:
        di = DI_VALUES

    # Extract data from precipitation script
    precipitation_values_large = np.load('processed_data/MONTHLY_PRECIP_V2.npy')[:, ::-1, :]  # flip latitude for precip
    precipitation_lat = np.load('processed_data/LAT_PRECIP.npy')[::-1]
    precipitation_lon = np.load('processed_data/LON_PRECIP.npy')

    # Extract data from temperature script, does not need to be downsampled
    min_temp = np.load('processed_data/MONTHLY_MINTEMP_V2.npy')
    max_temp = np.load('processed_data/MONTHLY_MAXTEMP_V2.npy')
    temperature_lat = np.load('processed_data/LAT_TEMP.npy')
    temperature_lon = np.load('processed_data/LON_TEMP.npy')

    # Extract data from soil moisture script
    sm_values_large = concat_sm()
    sm_lat = np.load('processed_data/LAT_SMAP_2015.npy')
    sm_lon = np.load('processed_data/LON_SMAP_2015.npy')

    # Downsample input variables to fit drought index resolution
    precip = downsample(precipitation_lat, precipitation_lon, precipitation_values_large, 'sum')
    sm = downsample(sm_lat, sm_lon, sm_values_large, 'avg')

    assert precip.shape == min_temp.shape == max_temp.shape == sm.shape

    x, y = build_x_y(di, precip, min_temp, max_temp, sm)

    return y, x


def concat_sm():
    """
    Concatenate various soil moisture arrays into one.
    """
    sm_2015 = np.load('processed_data/MONTHLY_VALUES_SMAP_2015.npy')
    sm_2016 = np.load('processed_data/MONTHLY_VALUES_SMAP_2016.npy')
    sm_2017 = np.load('processed_data/MONTHLY_VALUES_SMAP_2017.npy')
    sm = np.concatenate((sm_2015, sm_2016, sm_2017), axis=0)
    return sm


def downsample(var_lat, var_lon, var_values_large, method):
    """
    Downsamples data spatially to align with drought index scale. Ignores nans by using
    nansum and nanmean.

    Inputs:
        Method: either 'sum' or 'avg' the values being downsampled
    """
    grid_size = DI_LAT[0] - DI_LAT[1]
    values_array = np.zeros((var_values_large.shape[0], len(DI_LAT), len(DI_LON)))
    for time in range(var_values_large.shape[0] - 1):
        for lat_idx in range(len(DI_LAT)):
            for lon_idx in range(len(DI_LON)):
                _, fine_lat_idx = get_area_coords(var_lat, [(DI_LAT[lat_idx] - 0.5 * grid_size),
                                                            (DI_LAT[lat_idx] + 0.5 * grid_size)])
                _, fine_lon_idx = get_area_coords(var_lon, [(DI_LON[lon_idx] - 0.5 * grid_size),
                                                            (DI_LON[lon_idx] + 0.5 * grid_size)])
                if method == 'sum':
                    value = np.nansum(var_values_large[time, fine_lat_idx[0]:fine_lat_idx[1],
                                      fine_lon_idx[1]:fine_lon_idx[0]], axis=None)
                else:  # method = 'avg'
                    value = np.nanmean(var_values_large[time, fine_lat_idx[0]:fine_lat_idx[1],
                                       fine_lon_idx[1]:fine_lon_idx[0]], axis=None)
                values_array[time, lat_idx, lon_idx] = float(value)
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
    **Note: Used for linear regression model but not many to one RNN.

    Build y array from drought index array to input into model.
    Each lat, lon, during one year, corresponds to one training example.
    """
    m = di_values.shape[0] * di_values.shape[1] * di_values.shape[2]  # number of training examples
    y = {}
    for year_idx in range(0, di_values.shape[0]):
        new_vals = np.reshape(di_values[year_idx, :, :], (1, di_values.shape[1] * di_values.shape[2]))
        y[year_idx] = new_vals
    y = np.concatenate([y[year_idx] for year_idx in range(0, di_values.shape[0])], axis=1)
    y = y.T
    assert y.shape == (m, 1)
    return y


def build_x(precip, max_temp, min_temp, sm):
    """
    **Note: Used for linear regression model but not many to one RNN.
    """
    m = DI_VALUES.shape[0] * DI_VALUES.shape[1] * DI_VALUES.shape[2]  # number of training examples
    x = {}
    grid_area = DI_VALUES.shape[1] * DI_VALUES.shape[2]
    for year_idx in range(0, DI_VALUES.shape[0]):
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


def build_x_y(di, precip, min_temp, max_temp, sm):
    """
    Builds the x and y arrays to be used in time series model. Takes out any training examples
    with NaN in any of the inputs or y matrix. Creates array by going through every lat lon
    value at each corresponding year.
    """
    m = 0
    x = np.empty((0, 6, NUM_VARS))
    y = np.empty((0, 1))
    for year_idx in range(di.shape[0] - 1):
        for lat_idx in range(di.shape[1]):
            for lon_idx in range(di.shape[2]):
                x_m = np.zeros((1, 6, NUM_VARS))
                x_m[:, :, 0] = precip[6 * year_idx:6 * (year_idx + 1), lat_idx, lon_idx]
                x_m[:, :, 1] = min_temp[6 * year_idx:6 * (year_idx + 1), lat_idx, lon_idx]
                x_m[:, :, 2] = max_temp[6 * year_idx:6 * (year_idx + 1), lat_idx, lon_idx]
                x_m[:, :, 3] = sm[6 * year_idx:6 * (year_idx + 1), lat_idx, lon_idx]
                y_m = di[year_idx, lat_idx, lon_idx].reshape(1, 1)
                if not np.isnan(x_m).any():  # no nans in x matrix
                    if not np.isnan(y_m):  # no nans in y matrix
                        m += 1
                        x = np.concatenate((x, x_m), axis=0)
                        y = np.concatenate((y, y_m), axis=0)
    print("Number of training examples: " + str(m))
    assert x.shape[0] == y.shape[0] == m
    return x, y


def drop_nans_old(x_nans, y_nans):
    """
    **Note: Used for linear regression model but not many to one RNN.

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
outfile = TemporaryFile()
# np.save('Y',Y)  # first test
# np.save('X',X)  # first test
np.save('Y_v2', Y)  # version 2
np.save('X_v2', X)  # version 2
