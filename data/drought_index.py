"""
Authors: Bennett Bolen, Hannah Hampson, Mo Sodwatana
February 24, 2021
"""

# TODO: downscale temporally extracting correct 6 month periods corresponding to our years

import netCDF4 as nc
from variables import *  # imports variables such as lat_lims, lon_lims


def main():
    filename = 'spei06.nc'
    # view_data(filename)
    total_values, lat, lon, time, values = extract_data(filename)
    print(values.shape)


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
    time = data.variables['time'][:]  # days since 1901-01-01
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    area_lat, idx_lat = get_area_coords(lat, lat_lims[::-1]) # reverse latitide limits
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


if __name__ == '__main__':
    main()