import numpy as np
import h5py
import os
from os import listdir
import numpy.ma as ma
from os.path import isfile, join

global combined_data

def build_sm_array():
    # Builds the soil moisture dtaset from the individual data files. The .h5 files should be located in the folder
    # specified by rootdir, with data from each year in a separate subfolder.

    rootdir = os.getcwd() + "/"
    #rootdir = "C:/Users/Bennett/Documents/School/Stanford/4. CS230 Project/Test Data/"
    print(rootdir)

    dirs = listdirs(rootdir)
    dirs = sorted(dirs)
    print(dirs)

    if not dirs:
        trimmed_data, trimmed_lats, trimmed_lons = extract_one_year(rootdir)

    else:
        print("hi")
        for idx in range(len(dirs)):

            if idx == 0:
                trimmed_data, trimmed_lats, trimmed_lons = extract_one_year(rootdir + dirs[idx])
                print(trimmed_data.shape)
            else:
                trimmed_data_temp, trimmed_lats_temp, trimmed_lons_temp = extract_one_year(rootdir + dirs[idx])
                trimmed_data = np.concatenate((trimmed_data, trimmed_data_temp))

                # trimmed_lats = np.concatenate((trimmed_lats, trimmed_lats_temp), axis=1)
                # trimmed_lons = np.concatenate((trimmed_lons, trimmed_lons_temp))
                #
                # if np.sum(trimmed_lats[:, idx-1] != trimmed_lats[:, idx]):
                #     raise Exception("Latitudes don't match. File Number: " + str(idx+1))
                #
                # if np.sum(trimmed_lons[idx-1, :] != trimmed_lons[idx, :]):
                #     raise Exception("Longitudes don't match. File Number" + str(idx+1))
            idx += 1

    return trimmed_data, trimmed_lats, trimmed_lons, dirs


def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def extract_one_year(path):
    # Extracts all the .h5 files in one folder and combines them to an averaged output for that year
    filenames = list_h5_files(path)

    #full_data, averaged_data, lats, lons = combine_arrays(filenames, path)
    averaged_data, lats, lons = combine_arrays(filenames, path)

    # lat_lims = [42, 31]  # latitude range over California
    # lon_lims = [-125, -114]  # longitude range over California
    # lat_coords, lat_idx = get_area_coords(lats, lat_lims)
    # long_coords, lon_idx = get_area_coords(lons, lon_lims)


    #
    # trimmed_data, trimmed_lats, trimmed_lons = trim_data(lat_lims, lon_lims, averaged_data, lats, lons)
    #
    # trimmed_data = np.reshape(trimmed_data, (1, trimmed_data.shape[0], trimmed_data.shape[1]))
    # trimmed_lats = np.reshape(trimmed_lats, (trimmed_lats.shape[0], 1))
    # trimmed_lons = np.reshape(trimmed_lons, (1, trimmed_lons.shape[0]))

    return averaged_data, lats, lons


def list_h5_files(path='.'):
    # Returns a list of all .h5 files in the directory specified by path

    h5_files = [f for f in listdir(path) if f.endswith(".h5")]

    return h5_files


def combine_arrays(filenames, path):
    # Combines the data of all the files stored in the list filenames into numpy arrays. Outputs are a combined array
    # which includes each individual data file, and an averaged array which is an average of all the files. Input
    # arrays must be of the same dimensions

    # lat_lims = [42, 31]  # latitude range over California
    # lon_lims = [-125, -114]  # longitude range over California
    # lat_coords, lat_idx = get_area_coords(lats, lat_lims)
    # long_coords, lon_idx = get_area_coords(lons, lon_lims)

    for i in range(len(filenames)):
        print(i)
        if i == 0:
            [data_temp, lats, lons] = import_file((path + "/" + filenames[i]))
            lat_lims = [49, 32.5]  # latitude range over California
            lon_lims = [-125, -114]  # longitude range over California
            lat_coords, lat_idx = get_area_coords(lats, lat_lims)
            long_coords, lon_idx = get_area_coords(np.transpose(lons), lon_lims)

            trimmed_data, trimmed_lats, trimmed_lons = trim_data(lat_idx, lon_idx, data_temp, lats, lons)
            trimmed_data = np.reshape(trimmed_data, (1, trimmed_data.shape[0], trimmed_data.shape[1]))
            trimmed_lats = np.reshape(trimmed_lats, (trimmed_lats.shape[0], 1))
            trimmed_lons = np.reshape(trimmed_lons, (1, trimmed_lons.shape[0]))

            combined_data = np.zeros((len(filenames), trimmed_data.shape[1], trimmed_data.shape[2]), dtype=np.float32)


        else:
            [data_temp, lats_temp, lons_temp] = import_file((path + "/" + filenames[i]))

            trimmed_data, trimmed_lats, trimmed_lons = trim_data(lat_idx, lon_idx, data_temp, lats, lons)
            trimmed_data = np.reshape(trimmed_data, (1, trimmed_data.shape[0], trimmed_data.shape[1]))
            lats_temp = np.reshape(trimmed_lats, (trimmed_lats.shape[0], 1))
            lons_temp = np.reshape(trimmed_lons, (1, trimmed_lons.shape[0]))

            # if np.sum(lats != lats_temp):
            #     raise Exception("Latitudes don't match. File Number: " + str(i+1))
            #
            # if np.sum(lons != lons_temp):
            #     raise Exception("Longitudes don't match. File Number" + str(i+1))

            trimmed_lats = lats_temp
            trimmed_lons = lons_temp


        combined_data[i, :, :] = trimmed_data
        print(combined_data.shape)
    print(combined_data.shape)
    weights = combined_data > -100

    averaged_data = np.ma.average(combined_data, weights=weights, axis=0)
    averaged_data = np.reshape(averaged_data, (1, averaged_data.shape[0], averaged_data.shape[1]))

    averaged_data = averaged_data.filled(fill_value=-9999)

    # return combined_data, averaged_data, lats[:, 0], lons[0, :]
    return averaged_data, trimmed_lats[:, 0], trimmed_lons[0, :]


def import_file(filename):
    # Imports a .h5 file and extracts the surface level soil moisture data, the latitudes, and the longitudes
    with h5py.File(filename, "r") as f:
        data = f["Forecast_Data"]["sm_surface_forecast"][:, :]
        data = np.reshape(data, (1, data.shape[0], data.shape[1]))

        lats = f["cell_lat"][:, 0]
        lats = np.reshape(lats, (lats.shape[0], 1))

        lons = f["cell_lon"][0, :]
        lons = np.reshape(lons, (1, lons.shape[0]))

        # lat_lims = [42, 31]  # latitude range over California
        # lon_lims = [-125, -114]  # longitude range over California

        # trimmed_data, trimmed_lats, trimmed_lons = trim_data(lat_idx, lon_idx, data, lats, lons)
        #
        # trimmed_data = np.reshape(trimmed_data, (1, trimmed_data.shape[0], trimmed_data.shape[1]))
        # trimmed_lats = np.reshape(trimmed_lats, (trimmed_lats.shape[0], 1))
        # trimmed_lons = np.reshape(trimmed_lons, (1, trimmed_lons.shape[0]))

        return data, lats, lons


def trim_data(lat_idx, lon_idx, area, lats, lons):
    # Trims data arrays to the specified latitude and longitude limits
    # lat_coords, lat_idx = get_area_coords(lats, lat_lim)
    # long_coords, lon_idx = get_area_coords(lons, lon_lim)

    trimmed_data = area[0, lat_idx[0]:lat_idx[1], lon_idx[0]:lon_idx[1]]
    trimmed_lats = lats[lat_idx[0]:lat_idx[1]]
    trimmed_lons = lons[lon_idx[0]:lon_idx[1]]

    return trimmed_data, trimmed_lats, trimmed_lons


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


SIX_MONTH_VALUES, LAT, LON, YEARS = build_sm_array()


np.save("SIX_MONTH_VALUES_2years.npy", SIX_MONTH_VALUES)
np.save("LAT_2years.npy", LAT)
np.save("LON_2years.npy", LON)


