import numpy as np
import dask

def search_ranges(range_record, trajectories, range_id='Licence', trajectory_id='plate',
                  begin_time='begin_time', end_time='end_time'):
    _id = range_record[range_id]
    _s_t, _e_t = range_record[begin_time], range_record[end_time]
    _id_s = trajectories[trajectory_id].searchsorted(_id, side='left')
    _id_e = trajectories[trajectory_id].searchsorted(_id, side='right')
    _temp = trajectories['timestamp'].iloc[_id_s: _id_e]
    _s_i = _id_s + _temp.searchsorted(_s_t, side='left')
    _e_i = _id_s + _temp.searchsorted(_e_t, side='right')
    return _s_i, _e_i

def haversine_dask(lng1, lat1, lng2, lat2, miles=False):
    AVG_EARTH_RADIUS = 6371008.8  # in meter
    lng1, lat1, lng2, lat2 = map(dask.array.radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    d = dask.array.sin(dlat * 0.5) ** 2 + dask.array.cos(lat1) * dask.array.cos(lat2) * dask.array.sin(dlon * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * dask.array.arcsin(dask.array.sqrt(d))
    return h  # in meter

def find_nearest_station(traj, cs_info=None):
    from sklearn.metrics.pairwise import haversine_distances
    dis = haversine_distances(np.radians(traj), np.radians(cs_info))
    return dis.argmin(axis=1), dis.min(axis=1)

def coord2radian(lat=None, lng=None, lat_first=True):
    _lat_rad, _lng_rad = np.radians(lat).to_numpy(), np.radians(lng).to_numpy()
    if lat_first:
        _rad = np.concatenate((_lat_rad[:, np.newaxis], _lng_rad[:, np.newaxis]), axis=1)
    else:
        _rad = np.concatenate((_lng_rad[:, np.newaxis], _lat_rad[:, np.newaxis]), axis=1)
    return _rad
