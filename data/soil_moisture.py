import numpy as np
import h5py
from os import listdir
from os.path import isfile, join


def main():
    filenames = list_files()

    full_data, averaged_data, lats, lons = combine_arrays(filenames)

    lat_lims = [42, 31]  # latitude range over California
    lon_lims = [-125, -114]  # longitude range over California

    global trimmed_data_SM, trimmed_lats_SM, trimmed_lons_SM
    trimmed_data_SM, trimmed_lats_SM, trimmed_lons_SM = trim_data(lat_lims, lon_lims, averaged_data, lats, lons)


def list_files(path='.'):
    h5_files = [f for f in listdir(path) if (isfile(f) and f.endswith(".h5"))]

    return h5_files


def combine_arrays(filenames):
    for i in range(len(filenames)):
        if i == 0:
            [combined_data, lats, lons] = import_file(filenames[i])

        else:
            [surface, lats_temp, lons_temp] = import_file(filenames[i])
            combined_data = np.concatenate((combined_data, surface))
            lats = np.concatenate((lats, lats_temp), axis=1)
            lons = np.concatenate((lons, lons_temp))

            if np.sum(lats[:, i-1] != lats[:, i]):
                raise Exception("Latitudes don't match. File Number: " + str(i+1))

            if np.sum(lons[i-1, :] != lons[i, :]):
                raise Exception("Longitudes don't match. File Number" + str(i+1))

    averaged_data = np.mean(combined_data, axis=0)
    averaged_data = np.reshape(averaged_data, (1, averaged_data.shape[0], averaged_data.shape[1]))

    return combined_data, averaged_data, lats[:, 0], lons[0, :]


def import_file(filename):
    with h5py.File(filename, "r") as f:
        surface = f["Forecast_Data"]["sm_surface_forecast"][:, :]
        surface = np.reshape(surface, (1, surface.shape[0], surface.shape[1]))

        lats = f["cell_lat"][:, 0]
        lats = np.reshape(lats, (lats.shape[0], 1))

        lons = f["cell_lon"][0, :]
        lons = np.reshape(lons, (1, lons.shape[0]))

        return surface, lats, lons


def trim_data(lat_lim, lon_lim, area, lats, lons):
    lat_coords, lat_idx = get_area_coords(lats, lat_lim)
    long_coords, lon_idx = get_area_coords(lons, lon_lim)

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


if __name__ == '__main__':
    main()
