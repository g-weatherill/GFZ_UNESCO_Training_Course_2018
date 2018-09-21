"""
Scripts for running the Gruenthal declustering algorithm
"""
import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize, LogNorm

def convert_catalogue_to_dict(data):
    """
    Reads a hmtk formatted catalogue and returns as an ordered dict,
    parsing the dates and times to a numpy.datetime64 array
    """
    headers = [col for col in data.columns]
    # Sort into arrays
    catalogue = OrderedDict([(header, data[header].values)
                             for header in headers])
    # Build the datetime objects
    dtimes = []
    for i in range(data.shape[0]):
        dtimes.append("{:s}-{:s}-{:s}T{:s}:{:s}:{:s}".format(
            str(catalogue["year"][i]),
            str(catalogue["month"][i]).zfill(2),
            str(catalogue["day"][i]).zfill(2),
            str(catalogue["hour"][i]).zfill(2),
            str(catalogue["minute"][i]).zfill(2),
            "{0:07.4f}".format(catalogue["second"][i])))
    catalogue["dtime"] = np.array(dtimes, dtype="datetime64")
    return catalogue

def read_catalogue(filename):
    data = pd.read_csv(filename, sep=",")
    return convert_catalogue_to_dict(data)


def haversine(lon1, lat1, lon2, lat2, radians=False, earth_rad=6371.227):
    """
    Allows to calculate geographical distance
    using the haversine formula.

    :param lon1: longitude of the first set of locations
    :type lon1: numpy.ndarray
    :param lat1: latitude of the frist set of locations
    :type lat1: numpy.ndarray
    :param lon2: longitude of the second set of locations
    :type lon2: numpy.float64
    :param lat2: latitude of the second set of locations
    :type lat2: numpy.float64
    :keyword radians: states if locations are given in terms of radians
    :type radians: bool
    :keyword earth_rad: radius of the earth in km
    :type earth_rad: float
    :returns: geographical distance in km
    :rtype: numpy.ndarray
    """
    if not radians:
        cfact = np.pi / 180.
        lon1 = cfact * lon1
        lat1 = cfact * lat1
        lon2 = cfact * lon2
        lat2 = cfact * lat2

    # Number of locations in each set of points
    if not np.shape(lon1):
        nlocs1 = 1
        lon1 = np.array([lon1])
        lat1 = np.array([lat1])
    else:
        nlocs1 = np.max(np.shape(lon1))
    if not np.shape(lon2):
        nlocs2 = 1
        lon2 = np.array([lon2])
        lat2 = np.array([lat2])
    else:
        nlocs2 = np.max(np.shape(lon2))
    # Pre-allocate array
    distance = np.zeros((nlocs1, nlocs2))
    i = 0
    while i < nlocs2:
        # Perform distance calculation
        dlat = lat1 - lat2[i]
        dlon = lon1 - lon2[i]
        aval = (np.sin(dlat / 2.) ** 2.) + (np.cos(lat1) * np.cos(lat2[i]) *
                                            (np.sin(dlon / 2.) ** 2.))
        distance[:, i] = (2. * earth_rad * np.arctan2(np.sqrt(aval),
                                                      np.sqrt(1 - aval))).T
        i += 1
    return distance


def distance_window(mag):
    """
    Returns the distance window (in km) for the Gruenthal methodology
    """
    return np.exp(1.77474 + np.sqrt(0.03656 + 1.02237 * mag))


def foreshock_time_window(mag):
    """
    Returns the foreshock time window (days) for the Gruenthal methodology
    """
    window = np.zeros_like(mag)
    idx = mag < 7.784
    window[idx] = np.exp(-4.76590 + np.sqrt(0.61937 + 17.32149 * mag[idx]))
    idx = np.logical_not(idx)
    window[idx] = np.exp(6.44307 + 0.05515 * mag[idx])
    return window


def aftershock_time_window(mag):
    """
    Returns the aftershock time window (days) for the Gruenthal methodology
    """
    window = np.zeros_like(mag)
    idx = mag < 6.643
    window[idx] = np.exp(-3.94655 + np.sqrt(0.61937 + 17.32149 * mag[idx]))
    idx = np.logical_not(idx)
    window[idx] = np.exp(6.44307 + 0.05515 * mag[idx])
    return window


def gruenthal_declustering(catalogue, verbose=False):
    """
    """
    # Create numerical ids with original event order
    nids = np.arange(0, len(catalogue["eventID"]))
    # Sort into order of descending magnitude
    sidx = np.argsort(catalogue["magnitude"])[::-1]
    lons, lats, depths, mags, dtimes, ids = (
        catalogue["longitude"][sidx], catalogue["latitude"][sidx],
        catalogue["depth"][sidx], catalogue["magnitude"][sidx],
        catalogue["dtime"][sidx], nids[sidx]
    )
    # Get time and distance windows
    rwin = distance_window(mags)
    # Time windows returned in terms of days, round up to nearest integer
    fswin = np.ceil(foreshock_time_window(mags)).astype(int)
    fswin = fswin.astype("timedelta64[D]")
    aswin = np.ceil(aftershock_time_window(mags)).astype(int)
    aswin = aswin.astype("timedelta64[D]")
    neqs = len(mags)  # Number of events
    vcl = np.zeros(neqs, dtype=int)  # Set cluster indices to 0
    cluster_counter = 0
    for i in range(neqs):
        d_t = (dtimes - dtimes[i]).astype("timedelta64[D]")
        # Find events within forshock and aftershock time window
        t_idx = np.logical_and(d_t >= -fswin[i],
                               d_t <= aswin[i])
        if np.sum(t_idx) <= 1:
            # No events within fore- or aftershock window of this event
            continue
        # Check distance
        rval = haversine(lons[t_idx], lats[t_idx], lons[i], lats[i]).flatten()
        rval = np.sqrt(rval ** 2. + (depths[t_idx] - depths[i]) ** 2.)
        r_idx = rval <= rwin[i]
        if np.sum(r_idx) <= 1:
            # No events within distance window of this event
            continue
        t_idx[t_idx] = np.copy(r_idx)
        if verbose:
            print("Event %g [cluster %g ] (%s - %6.3f %6.3f %6.3f, M %.1f) has %g non-poisson events" % (
                  i, vcl[i], str(dtimes[i]), lons[i], lats[i], depths[i],
                  mags[i], np.sum(t_idx)))
        if vcl[i]:
            # Considered event was already in a cluster, so new
            # events extend the existing cluster
            vcl[np.logical_or(t_idx, vcl == vcl[i])] = vcl[i]
        else:
            # Is a new cluster
            cluster_counter += 1
            vcl[t_idx] = cluster_counter
            
    # Sort vcl back into original catalogue order
    vcl = vcl[np.argsort(ids)]
    new_vcl = np.zeros_like(vcl, dtype=int)
    flag_vector = np.zeros(neqs, dtype=int)
    # Loop through clusters and separate foreshock/mainshock/aftershock
    cluster_counter = 1
    for j in range(1, np.max(vcl)):
        idx = np.where(vcl == j)[0]
        if len(idx) <= 1:
            continue
        new_vcl[idx] = cluster_counter
        local_flag = np.zeros(len(idx), dtype=int)
        local_time = catalogue["dtime"][idx]
        local_mag = catalogue["magnitude"][idx]
        max_m = np.argmax(local_mag)
        local_dtime = (local_time - local_time[max_m]).astype(int)
        local_flag[local_dtime > 0] = 1  # Aftershock
        local_flag[local_dtime < 0] = -1 # Mainshock
        flag_vector[idx] = local_flag
        cluster_counter += 1
    return vcl, flag_vector


def export_catalogue_dataseries(filename, data, vcl, flagvector, purge=False):
    """
    Exports the catalogue to a file including the cluster flags

    If purge is True then returns only the mainshocks
    """
    # Add vcl and flagvector
    data["vcl"] = pd.Series(vcl)
    data["flagvector"] = pd.Series(flagvector)
    if purge:
        # Keep only mainshocks
        data = data[flagvector==0]
    data.to_csv(filename, sep=",", index=False)
    print("Exported to file %s" % filename)


def plot_cluster_sequence(cluster_id, catalogue, vcl, flagvector,
                          filename=None, filetype="png", dpi=300):
    """
    Plots the temporal and spatial distribution of the cluster
    """
    idx = vcl == cluster_id
    if not np.any(vcl):
        print("Cluster ID %g has no events!" % cluster_id)
    mainshock = np.logical_and(idx, flagvector==0)
    fig = plt.figure(figsize=(6,12))
    ax1 = fig.add_subplot(211)
    # Plot the time of the events
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    ax1.plot(catalogue["dtime"][idx], catalogue["magnitude"][idx], "b.")
    ax1.plot(catalogue["dtime"][mainshock],
             catalogue["magnitude"][mainshock], "ks")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Mw")
    ax1.grid(True)
    # format the ticks
    ax1.xaxis.set_major_locator(years)
    ax1.xaxis.set_major_formatter(yearsFmt)
    ax1.xaxis.set_minor_locator(months)
    ax1.format_xdata = mdates.DateFormatter('%Y-%m')
    
    fig.autofmt_xdate()
    ax2 = fig.add_subplot(212)
    d_t = (catalogue["dtime"][idx] -
           catalogue["dtime"][mainshock]).astype("timedelta64[D]")
    cax = ax2.scatter(catalogue["longitude"][idx], catalogue["latitude"][idx],
                      s=catalogue["magnitude"][idx] ** 2., c=d_t, marker="o")
    
    ax2.scatter(catalogue["longitude"][mainshock],
                catalogue["latitude"][mainshock],
                s=40, color="k", marker="s")
    ax2.grid(True)
    fig.colorbar(cax)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    if filename:
        plt.savefig(filename, format=filetype, dpi=dpi, bbox_inches="tight")


#def gruenthal_declustering_v2(catalogue):
#    """
#    """
#    # Create numerical ids with original event order
#    nids = np.arange(0, len(catalogue["eventID"]))
#    # Sort into order of descending magnitude
#    sidx = np.argsort(catalogue["magnitude"])[::-1]
#    print(catalogue["magnitude"][sidx])
#    lons, lats, depths, mags, dtimes, ids = (
#        catalogue["longitude"][sidx], catalogue["latitude"][sidx],
#        catalogue["depth"][sidx], catalogue["magnitude"][sidx],
#        catalogue["dtime"][sidx], nids[sidx]
#    )
#    # Get time and distance windows
#    rwin = distance_window(mags)
#    # Time windows returned in terms of days, round up to nearest integer
#    fswin = np.ceil(foreshock_time_window(mags)).astype(int)
#    fswin = fswin.astype("timedelta64[D]")
#    aswin = np.ceil(aftershock_time_window(mags)).astype(int)
#    aswin = aswin.astype("timedelta64[D]")
#    print(rwin)
#    print(fswin)
#    print(aswin)
#    neqs = len(mags)  # Number of events
#    vcl = np.zeros(neqs, dtype=int)  # Set cluster indices to 0
#    cluster_counter = 1
#    for i in range(neqs):
#        if vcl[i]:
#            # Event is already part of a cluster - skip
#            continue
#        d_t = (dtimes - dtimes[i]).astype("timedelta64[D]")
#        # Find events within forshock and aftershock time window
#        t_idx = np.logical_and(d_t >= -fswin[i],
#                               d_t <= aswin[i])
#        if np.sum(t_idx) <= 1:
#            # No events within fore- or aftershock window of this event
#            continue
#        # Check distance
#        rval = haversine(lons[t_idx], lats[t_idx], lons[i], lats[i]).flatten()
#        rval = np.sqrt(rval ** 2. + (depths[t_idx] - depths[i]) ** 2.)
#        r_idx = rval <= rwin[i]
#        if np.sum(r_idx) <= 1:
#            # No events within distance window of this event
#            continue
#        t_idx[t_idx] = np.copy(r_idx)
#        #print(rwin[i], aswin[i], -fswin[i])
#        #print(rval, d_t[t_idx])
#        #print(lons[i], lats[i], dtimes[i], lons[t_idx], lats[t_idx], dtimes[t_idx])
#        #breaker = here
#        vcl[t_idx] = cluster_counter
#        cluster_counter += 1        
#    # Sort vcl back into original catalogue order
#    vcl = vcl[np.argsort(ids)]
#    new_vcl = np.zeros_like(vcl, dtype=int)
#    flag_vector = np.zeros(neqs, dtype=int)
#    # Loop through clusters and separate foreshock/mainshock/aftershock
#    cluster_counter = 1
#    for j in range(1, np.max(vcl)):
#        idx = np.where(vcl == j)[0]
#        if len(idx) <= 1:
#            continue
#        new_vcl[idx] = cluster_counter
#        local_flag = np.zeros(len(idx), dtype=int)
#        local_time = catalogue["dtime"][idx]
#        local_mag = catalogue["magnitude"][idx]
#        max_m = np.argmax(local_mag)
#        local_dtime = (local_time - local_time[max_m]).astype(int)
#        local_flag[local_dtime > 0] = 1  # Aftershock
#        local_flag[local_dtime < 0] = -1 # Mainshock
#        flag_vector[idx] = local_flag
#        cluster_counter += 1
#    return vcl, flag_vector
#
#def plot_cluster_sequence(cluster_id, catalogue, vcl, flagvector):
#    idx = vcl == cluster_id
#    if not np.any(vcl):
#        print("Cluster ID %g has no events!" % cluster_id)
#    mainshock = np.logical_and(idx, flagvector==0)
#    fig = plt.figure(figsize=(6,12))
#    ax1 = fig.add_subplot(211)
#    # Plot the time of the events
#    years = mdates.YearLocator()   # every year
#    months = mdates.MonthLocator()  # every month
#    yearsFmt = mdates.DateFormatter('%Y')
#    ax1.plot(catalogue["dtime"][idx], catalogue["magnitude"][idx], "b.")
#    ax1.plot(catalogue["dtime"][mainshock], catalogue["magnitude"][mainshock], "ks")
#    ax1.set_xlabel("Time")
#    ax1.set_ylabel("Mw")
#    ax1.grid(True)
#    # format the ticks
#    ax1.xaxis.set_major_locator(years)
#    ax1.xaxis.set_major_formatter(yearsFmt)
#    ax1.xaxis.set_minor_locator(months)
#
#    ax1.format_xdata = mdates.DateFormatter('%Y-%m')
#    fig.autofmt_xdate()
#    ax2 = fig.add_subplot(212)
#    d_t = (catalogue["dtime"][idx] - catalogue["dtime"][mainshock]).astype("timedelta64[D]")
#    cax = ax2.scatter(catalogue["longitude"][idx], catalogue["latitude"][idx],
#                s=catalogue["magnitude"][idx] ** 2., c=d_t, marker="o")
#    
#    ax2.scatter(catalogue["longitude"][mainshock], catalogue["latitude"][mainshock],
#                      s=40, color="k", marker="s")
#    ax2.grid(True)
#    fig.colorbar(cax)
#    ax2.set_xlabel("Longitude")
#    ax2.set_ylabel("Latitude")
