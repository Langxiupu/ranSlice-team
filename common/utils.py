from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file
import numpy as np
from datetime import datetime, timedelta, timezone
import copy
import pickle
import os


def velocitytolaps(sat):
    laps_per_day = sat.model.no_kozai * 60 * 24 / (2*np.pi)
    return laps_per_day


def lapstoaltitude(laps):
    period = 86400.0 / laps

    coeff = 21613.546
    orbit_radius = coeff * np.power(period, 2/3)
    earth_radius = 6371.393e3
    altitude = (orbit_radius - earth_radius) / 1e3
    return altitude


def read_satellites(file_name="starlink.tle"):
    ts = load.timescale()
    with load.open(file_name) as f:
        # List has 6227 satellites
        satellites = list(parse_tle_file(f, ts))    
    print(f"The total satellites in constellation is {len(satellites)}")
    satellites = satellites[: 1782]

    satellites_filter1 = []
    for satellite in satellites:
        lap = velocitytolaps(satellite)
        altitude = lapstoaltitude(lap)
        if 545 <= altitude <= 580:
            satellites_filter1.append(satellite)
    satellites_filter2 = []
    for satellite in satellites_filter1:
        sat_inclo = np.rad2deg(satellite.model.inclo)
        if sat_inclo <= 60:
            satellites_filter2.append(satellite)
    print("The Starlink constellation consists of {} satellites".format(len(satellites_filter2)))
    return satellites_filter2


# support step for 1hour, 1min, 1sec
def generate_time_interval(start_time, end_time, h_step=0, m_step=0, s_step=0):
    start_time = datetime(*start_time, tzinfo=timezone.utc)
    end_time = datetime(*end_time, tzinfo=timezone.utc)
    if h_step:
        delta = timedelta(hours=h_step)
    elif m_step:
        delta = timedelta(minutes=m_step)
    elif s_step:
        delta = timedelta(seconds=s_step)
    
    cur_time = copy.deepcopy(start_time)
    time_list = []
    while cur_time < end_time:
        time_list.append(cur_time)
        cur_time += delta
    return time_list


def generate_T(start_time, end_time, h_step=0, m_step=0, s_step=0):
    dt_list = generate_time_interval(start_time, end_time, h_step, m_step, s_step)
    ts = load.timescale()
    T = ts.from_datetimes(dt_list)
    return T


def filter_tle_file(lines, satellites, skip_names=False):
    sat_cnt = 0
    sat_len = len(satellites)

    b0 = b1 = b''
    with open("data/filtered_starlink.tle", "wb") as f:
        for b2 in lines:
            if (b2.startswith(b'2 ') and len(b2) >= 69 and
                b1.startswith(b'1 ') and len(b1) >= 69):

                if not skip_names and b0:
                    b0 = b0.rstrip(b' \n\r')
                    if b0.startswith(b'0 '):
                        b0 = b0[2:]  # Spacetrack 3-line format
                    name = b0.decode('ascii')
                else:
                    name = None
                
                if name != satellites[sat_cnt].name:
                    continue
                    
                sat_cnt += 1
                f.write(b0+b'\n')
                f.write(b1)
                f.write(b2)

                b0 = b1 = b''
                if sat_cnt == sat_len:
                    break
            else:
                b0 = b1
                b1 = b2

# f_c--MHz BW--MHz  dist--km  P--dBm G--dbi
def calc_shannon_C(f_c, BW, dist, P, G=30, noise=-88.2):
    path_loss = 20 * np.log10(dist) + 20 * np.log10(f_c) + 32.45
    # print("path_loss: ", path_loss)
    SNR = P - path_loss - noise
    
    # shannon formula: C = B log2(1 + SNR)
    C = BW * np.log2(1 + 10 ** (SNR / 10))
    return C


class VisibleSat:
    def __init__(self, sat_id, name, alt, azi, range):
        self.sat_id = sat_id
        self.sat_name = name
        self.altitude = alt
        self.azimuth = azi
        self.range = range

    def __repr__(self):
        return (
            f"VisibleSat(sat_id={self.sat_id}, name={self.sat_name!r}, "
            f"alt={self.altitude:.2f}°, azi={self.azimuth:.2f}°, "
            f"range={self.range:.2f}km)"
        )


def generate_visibles(sat_filename, num_users, ds_size, alt_th=37.5,
                      h_step=0, m_step=0, s_step=0,
                      save_dir="",
                      zone_lat=(44, 46), zone_lon = (-96, -94),
                      start_time=(2024, 7, 21, 12, 45, 0),
                      end_time=(2024, 7, 21, 13, 45, 0),
                      seed=None, start_idx=0):
    assert h_step or m_step or s_step
    # generate time interval
    # default: 2024-7-21 12:45--2024-7-21 13:45
    t = generate_T(start_time, end_time, h_step, m_step, s_step)
    # set random seed 
    if seed is not None:
        np.random.seed(seed)

    # whether the save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # load satellites
    satellites = read_satellites(sat_filename)

    for n_case in range(start_idx, start_idx+ds_size):
        output_name = n_case
        # sample user position by uniform distribution
        lat = np.random.uniform(zone_lat[0], zone_lat[1], num_users)
        lon = np.random.uniform(zone_lon[0], zone_lon[1], num_users)
        users_coord = list(zip(lat, lon))
        users_pos = [wgs84.latlon(*coord) for coord in users_coord]

        
        # calculate sat's position related to users
        slot_size = len(t)
        visibles_t_u = []   # initialize
        for _ in range(slot_size):
            visible_user = {}
            for i in range(num_users):
                visible_user[i] = {}
            visibles_t_u.append(visible_user)
        for i, user in enumerate(users_pos):
            for k, sat in enumerate(satellites):
                alt, azi, dist = (sat.at(t) - user.at(t)).altaz()
                alt = alt.degrees
                visible_t = alt >= alt_th
                for j, v in enumerate(visible_t):
                    if v:
                        visibles_t_u[j][i][k] = VisibleSat(k, sat.name, alt[j], azi.degrees[j], dist.km[j])

        # save visibles_t_u
        visibles_dir = save_dir + f"random_users_{num_users}/" + "visibles_t_u/"
        os.makedirs(visibles_dir, exist_ok=True)
        if output_name is not None:
            name = visibles_dir + "visibles_t_u-{}.pkl".format(output_name)
        else:
            name = visibles_dir + "visibles_t_u-{}.pkl".format(num_users)
        with open(name, "wb") as f:
            pickle.dump(visibles_t_u, f)
        
        
        # save user pos
        user_pos_dir = save_dir + f"random_users_{num_users}/" +"user_pos/"
        os.makedirs(user_pos_dir, exist_ok=True)
        if output_name is not None:
            name = user_pos_dir + "users_pos-{}.pkl".format(output_name)
        else:
            name = user_pos_dir + "users_pos-{}.pkl".format(num_users)
        # save users' position  
        with open(name, "wb") as f:
            pickle.dump(users_pos, f)


    """
    calculate visible matrix list
    visible matrix shape: (num_users, num_sats)
    """
    # visible_matrix_list = []
    # for t in range(len(visibles_t_u)):
    #     visible_matrix = np.zeros((num_users, len(satellites)))
    #     for u in range(num_users):
    #         visible_matrix[u, list(visibles_t_u[t][u].keys())] = 1
    #     visible_matrix_list.append(visible_matrix)

    # # save visible matrix list
    # visible_matrix_dir = save_dir + "visible_matrix/"
    # os.makedirs(visible_matrix_dir, exist_ok=True)
    # with open(visible_matrix_dir+"visible_matrix_list-{}.pkl".format(num_users), "wb") as f:
    #     pickle.dump(visible_matrix_list, f)

    return 

    

def count_load(com_dict, visible_sats):
    load = {}
    for ele in visible_sats:
        load[ele] = 0
    for _, id in com_dict.items():
        if id == -1:
            continue
        load[id] += 1
    return list(load.values())

def count_max_visible(visible_matrix_list):
    max_satellite = -1
    for i in range(len(visible_matrix_list)):
        max_visible = np.sum(visible_matrix_list[i], axis=1).max()
        if max_visible > max_satellite:
            max_satellite = max_visible
    return max_satellite


def count_min_visible(visible_matrix_list):
    min_satellite = 1000
    for i in range(len(visible_matrix_list)):
        min_visible = np.sum(visible_matrix_list[i], axis=1).min()
        if min_visible < min_satellite:
            min_satellite = min_visible
    return min_satellite

"""
allo_t: (num_users, num_sats) HO decisions in slot t
com_stats: (num_users, num_sats) connection states in slot t-1
sc_s_stats: (num_sats, num_sc) SC states in slot t-1
sc_u_stats: (num_users, num_sc) SC states in slot t-1
"""
def update_com_stats(allo_t, com_stats, sc_s_stats, sc_u_stats, initialize=False):
    capacity = sc_s_stats.shape[1]
    # sat_id is indexed from 1
    SAT_IDX = np.arange(1, sc_s_stats.shape[0]+1).reshape(-1, 1)
    SC_IDX = np.arange(0, capacity).reshape(-1, 1)
    sat_targets = np.squeeze((allo_t @ SAT_IDX).astype(np.int32), axis=1)

    ho_n = 0
    ho_failure_n = 0
    if initialize:
        for u in range(allo_t.shape[0]):
            # satellite selection for user u
            com_target = sat_targets[u] - 1
            # if there is no connection
            if com_target == -1:
                continue
            
            # check if there is available SCs
            if np.sum(sc_s_stats[com_target]) < capacity:
                i = np.random.randint(0, capacity)
                while sc_s_stats[com_target, i] == 1:
                    i = np.random.randint(0, capacity)
                sc_s_stats[com_target, i] = 1
                sc_u_stats[u, i] = 1
                com_stats[u, com_target] = 1
        return ho_n, ho_failure_n
    else:
        # users connects satellite in random order
        USER_IDX = np.arange(0, allo_t.shape[0])
        np.random.shuffle(USER_IDX)
        for u in USER_IDX:
            com_target = sat_targets[u] - 1
            orig_target = (com_stats[u] @ SAT_IDX)[0] - 1

            # if there is no HO
            if com_target != -1 and com_target == orig_target:
                continue
            
            # HO occurs
            ho_n += 1
            # release the original connection, if exists
            if orig_target != -1:
                com_stats[u] = 0
                sc_i = (sc_u_stats[u] @ SC_IDX)[0]
                sc_u_stats[u, sc_i] = 0
                sc_s_stats[com_target, sc_i] = 0
            if com_target == -1:    # if there is no coonection, HO fails
                ho_n += 1
                ho_failure_n += 1
                continue
            
            # establish new connection through random access
            if np.sum(sc_s_stats[com_target]) < capacity:   # check if there is available SC
                # randomly select a SC
                i = np.random.randint(0, capacity)
                if sc_s_stats[com_target, i] == 1:
                    ho_failure_n += 1
                    continue
                # successfully establish new connection
                sc_s_stats[com_target, i] = 1
                sc_u_stats[u, i] = 1
                com_stats[u, com_target] = 1
            else:
                ho_failure_n += 1

        return ho_n, ho_failure_n
