"""
Script for plotting our input and drought index data for a single location through time.
"""

import numpy as np
from matplotlib import pyplot as plt


def main():
    precip, precip_lat, precip_lon, min_temp, max_temp, di, di_lat, di_lon = load_data()
    location_coords = [37.427474, -122.170277]  # stanford coordinates
    # plot_data(location_coords, precip, precip_lat, precip_lon)
    # plot_data(location_coords, di, di_lat, di_lon)

    precip_location = extract_location_data(precip, precip_lat, precip_lon, location_coords)
    di_location = extract_location_data(di, di_lat, di_lon, location_coords)
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    plt.scatter(np.arange(2015, 2019), di_location, color='red')
    plt.xlim([2015, 2018.5])
    plt.ylim([-2.5, 2.5])
    plt.ylabel('SPEI')
    ax = fig.add_subplot(2, 1, 2)
    plt.plot(precip_location)
    plt.xlabel('Month')
    plt.xlim([10, 48])
    plt.ylabel('Monthly Precip')
    ax.axvspan(11, 17, color='blue', alpha=0.3, label='Wet Season')
    ax.axvspan(17, 23, color='red', alpha=0.3, label='Dry Season')
    ax.axvspan(23, 29, color='blue', alpha=0.3)
    ax.axvspan(29, 35, color='red', alpha=0.3)
    ax.axvspan(35, 41, color='blue', alpha=0.3)
    ax.axvspan(41, 47, color='red', alpha=0.3)
    plt.legend()
    plt.show()


def load_data():
    precip = np.load('../processed_data/PRECIP_monthly.npy')
    precip_lat = np.load('../processed_data/LAT_PRECIP.npy')
    precip_lon = np.load('../processed_data/LON_PRECIP.npy')
    min_temp = np.load('../processed_data/MONTHLY_MINTEMP.npy')
    max_temp = np.load('../processed_data/MONTHLY_MAXTEMP.npy')
    di = np.load('../processed_data/SIX_MONTH_VALUES_DI.npy')
    di_lat = np.load('../processed_data/LAT_DI.npy')
    di_lon = np.load('../processed_data/LON_DI.npy')
    return precip, precip_lat, precip_lon, min_temp, max_temp, di, di_lat, di_lon


def extract_location_data(values_array, values_lat, values_lon, location_coords):
    idx = find_indices(values_lat, values_lon, location_coords)
    return values_array[:, idx[0], idx[1]]


def find_indices(values_lat, values_lon, location_coords):
    idx = [None, None]
    # find lat idx
    minimum = float("inf")
    for i in range(len(values_lat)):
        if abs(location_coords[0] - values_lat[i]) < minimum:
            final_value = i
            minimum = abs(location_coords[0] - values_lat[i])
    idx[0] = final_value
    # find lon idx
    minimum = float("inf")
    for i in range(len(values_lon)):
        if abs(location_coords[1] - values_lon[i]) < minimum:
            final_value = i
            minimum = abs(location_coords[1] - values_lon[i])
    idx[1] = final_value
    return idx


def plot_data(location_coords, values, values_lat, values_lon):
    values_location = extract_location_data(values, values_lat, values_lon, location_coords)
    plt.plot(values_location)
    plt.show()


if __name__ == '__main__':
    main()
