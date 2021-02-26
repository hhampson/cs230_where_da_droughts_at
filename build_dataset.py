"""
Authors: Bennett Bolen, Hannah Hampson, Mo Sodwatana
February 24, 2021

This script is for building the dataset.
"""

import numpy as np
from data import precipitation, drought_index


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

    # Extract data from temperature script

    # Extract data from soil moisture script

    # Downsample input variables to fit drought index resolution
    precip = downsample(precipitation_lat, precipitation_lon, precipitation_values_large, precipitation_years, 'sum')

    y = build_y(DI_VALUES)
    # x = build_x()
    return y


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
    y = np.zeros((1, 1))
    for year_idx in range(len(di_values.shape[0])):
        new_vals = np.reshape(di_values[year_idx, :, :], (1, di_values.shape[1] * di_values.shape[2]))
        y = np.concatenate([y, new_vals])
    y = y[1:, :, :]
    assert y.shape == (1, m)
    return y


def build_x(precip, max_temp, min_temp, soil_moisture):
    m = DI_VALUES.shape[0] * DI_VALUES.shape[1] * DI_VALUES.shape[2]  # number of training examples
    x = np.zeros(NUM_VARS, 1)
    grid_area = DI_VALUES.shape[1] * DI_VALUES.shape[2]
    for year_idx in range(len(DI_VALUES.shape[0])):
        precip_row = np.reshape(precip[year_idx, :, :], (1, grid_area))
        max_temp_row = np.reshape(max_temp[year_idx, :, :], (1, grid_area))
        min_temp_row = np.reshape(min_temp[year_idx, :, :], (1, grid_area))
        soil_moisture_row = np.reshape(soil_moisture[year_idx, :, :], (1, grid_area))
        new_vals = [precip_row, max_temp_row, min_temp_row, soil_moisture_row]
        x = np.concatenate([x, new_vals])
    x = x[1:, :, :]
    assert x.shape == (NUM_VARS, m)
    return x


Y = main()
