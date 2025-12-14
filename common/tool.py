import csv
import pickle
from turtle import position
import gym
from torch import Value
from commModule.UserTerminal import Position, UserTerminal
from commModule.UserTerminal import Distance
from commModule.leoSat import LEOSatellite
from commModule.mac_scheduler import MacScheduler
import random
import numpy as np

# 根据用户终端位置的CSV文件读取用户终端位置汇集成字典
def read_positions_from_csv(file_path):
    position_dict = {}
    index = 1  # 从1开始的键值

    with open(file_path, 'r', encoding='utf-8') as csvfile:
        # 创建CSV读取器，指定制表符分隔
        reader = csv.DictReader(csvfile, delimiter=',')

        for row in reader:
            try:
                # 从行数据中提取纬度和经度
                lat = float(row['latitude'])
                lon = float(row['longitude'])

                # 创建Position对象并添加到字典
                position_dict[index] = Position(latitude=lat, longitude=lon)
                index += 1  # 递增键值
            except (KeyError, ValueError) as e:
                # 处理可能的数据缺失或格式错误
                print(f"跳过无效行: {row}. 错误: {e}")
    return position_dict


# 根据给定文件路径和公共可见卫星,时隙以及用户数量获取特定时隙下每个用户终端与公共卫星之间的距离信息,以Dict{Dict}形式存储
def get_info_by_time(file_path, time, common_satellites, num_users):
    with open(file_path, "rb") as f:
        visibles_t_u = pickle.load(f)  # 读取可见卫星数据
    if time < 0 or time >= len(visibles_t_u):
        print(f"Error: Time index {time} is out of range.")
        return
    time_slot = visibles_t_u[time]
    distance_dict = {}
    for user in range(1, num_users + 1):
        visible_sats = time_slot[user - 1]  # 获取用户在该时隙可见的所有卫星
        user_distances = {}
        for sat in common_satellites:
            visible_sat = visible_sats[sat] if sat in visible_sats else None
            if visible_sat is not None:
                distance = Distance(altitude=visible_sat.altitude,
                                    azimuth=visible_sat.azimuth,
                                    Range=visible_sat.range)
                user_distances[sat] = distance
        distance_dict[user] = user_distances
    return distance_dict


# 根据文件路径获取时隙数量
def slot_num(file_path):
    with open(file_path, "rb") as f:
        visibles_t_u = pickle.load(f)  # 读取可见卫星数据
    return len(visibles_t_u)  # 返回时隙数量

"""
重新配置可用用户终端
参数：
    available_terminals: 可用用户终端
    unavailable_terminals: 不可用用户终端
    num_available_terminals: 可用用户终端数量
"""
def config_terminals(available_terminals, unavailable_terminals, num_available_terminals):
    # 首先可用用户终端数不能超过总用户终端数
    if num_available_terminals > len(available_terminals)+len(unavailable_terminals):
        print("Error: num_available_terminals cannot be greater than the total number of available terminals.")
        return
    # 其次判断可用用户终端字典大小与可用用户终端数量之间的关系
    if len(available_terminals) < num_available_terminals:
        # 从不可用用户终端中随机选择用户终端加入到可用用户终端中，同时从不可用用户终端中删除
        for i in range(num_available_terminals-len(available_terminals)):
            terminal = random.choice(list(unavailable_terminals.keys()))
            available_terminals[terminal] = unavailable_terminals.pop(terminal)
    elif len(available_terminals) > num_available_terminals:
        # 从可用用户终端中随机选择用户终端加入到不可用用户终端中，同时从可用用户终端中删除
        for i in range(len(available_terminals)-num_available_terminals):
            terminal = random.choice(list(available_terminals.keys()))
            unavailable_terminals[terminal] = available_terminals.pop(terminal)
            # 同时需要与终端的卫星断开连接
            unavailable_terminals[terminal].disconnect_sat()


def normalize_to_01(data):
    max_val = np.max(data)
    min_val = np.min(data)
    return (np.array(data) - min_val) / (max_val - min_val)


def calc_ratio_vec(start_t="13:00", end_t="14:00", mode="linear"):
    file_name = f'sat-data/traffic-wide-{start_t[:2]}pm.pkl'
    try: 
        with open(file_name, 'rb') as f:
            traffic = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_name} not found.")
    
    traffic = normalize_to_01(traffic)
    traffic_array = np.array(traffic)
    t = np.arange(12, dtype=float) * 300.0  # [0, 300, ..., 3300]

    t_new = np.arange(0.0, 3300.0 + 0.1, 0.1)
    if mode == "linear":
        traffic_new = np.interp(t_new, t, traffic_array)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented.")
    return traffic_new

def generate_sats_bd_rb_csv(satellite_ids, output_file):
    """
    生成包含指定卫星ID、带宽、SCS和RBs信息的CSV文件
    参数:
        satellite_ids: 卫星ID列表，例如[3743, 1666, 1644, 1143]
        output_file: 输出CSV文件的路径及名称，例如'sats-bd-scs-rb.csv'
    """
    # 带宽选项（单位：MHz），对应RB_LOOKUP_TABLE中的10e6和20e6
    # bandwidth_options = [10, 20]  
    bandwidth_options = [20]  
    # 每个带宽对应的SCS选项（单位：kHz），对应RB_LOOKUP_TABLE中的15e3、30e3、60e3
    scs_options = {
        # 10: [15, 30, 60],   # 10MHz带宽支持的SCS（kHz）
        # 20: [15, 30, 60]    # 20MHz带宽支持的SCS（kHz）
        20: [15]    # 20MHz带宽支持的SCS（kHz）
    }
    # 参考的RB查询表（键为（带宽Hz，SCS Hz），值为RBs）
    RB_LOOKUP_TABLE = {
        # 10MHz带宽
        (10e6, 15e3): 52,   # 10MHz, 15kHz SCS -> 52 RBs
        (10e6, 30e3): 24,   # 10MHz, 30kHz SCS -> 24 RBs
        (10e6, 60e3): 11,   # 10MHz, 60kHz SCS -> 11 RBs
        # 20MHz带宽
        (20e6, 15e3): 108,  # 20MHz, 15kHz SCS -> 108 RBs
        (20e6, 30e3): 51,   # 20MHz, 30kHz SCS -> 51 RBs
        (20e6, 60e3): 24,   # 20MHz, 60kHz SCS -> 24 RBs
    }

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # 写入表头：ID、Bandwidth(MHz)、SCS(kHz)、RBs
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Bandwidth(MHz)', 'SCS(kHz)', 'RBs'])
        
        for sat_id in satellite_ids:
            # 随机选择带宽（MHz）
            bandwidth_mhz = random.choice(bandwidth_options)
            # 转换带宽为Hz（匹配查询表的键）
            bandwidth_hz = bandwidth_mhz * 1e6
            
            # 根据带宽选择对应的SCS（kHz）
            scs_khz = random.choice(scs_options[bandwidth_mhz])
            # 转换SCS为Hz（匹配查询表的键）
            scs_hz = scs_khz * 1e3
            
            # 从查询表中获取RBs
            rbs = RB_LOOKUP_TABLE[(bandwidth_hz, scs_hz)]
            
            # 写入一行数据
            writer.writerow([sat_id, bandwidth_mhz, scs_khz, rbs])

    print(f"已成功生成{output_file}文件，包含{len(satellite_ids)}颗卫星的信息")

def read_sats_bd_rb_csv(file_path):
    """
    读取卫星CSV文件并直接返回以ID为键的字典（合并读取与字典构建步骤）
    参数:
        file_path: CSV文件的路径及名称，例如'sats-bd-scs-rb.csv'
    返回:
        dict: 以卫星ID为键的字典，结构为：
            {
                卫星ID: {
                    'Bandwidth(MHz)': int,  # 带宽（MHz）
                    'SCS(kHz)': int,        # 子载波间隔（kHz）
                    'RBs': int              # 资源块数
                },
                ...
            }
    异常:
        FileNotFoundError: 文件不存在时触发
        ValueError: 数据格式错误（表头不符、列数错误、类型转换失败等）
        Exception: 其他读取错误
    说明:
        - 若CSV中存在重复ID，后出现的记录会覆盖先出现的记录（建议确保ID唯一）
    """
    sat_id_dict = {}  # 最终返回的以ID为键的字典
    
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # 读取并验证表头
            header = next(reader)
            expected_header = ['ID', 'Bandwidth(MHz)', 'SCS(kHz)', 'RBs']
            if header != expected_header:
                raise ValueError(f"CSV表头格式错误，预期：{expected_header}，实际：{header}")
            
            # 读取数据行并构建字典
            for row_num, row in enumerate(reader, start=2):  # 行号从2开始（表头为1）
                # 校验列数
                if len(row) != 4:
                    raise ValueError(f"第{row_num}行数据列数错误，预期4列，实际{len(row)}列")
                
                # 转换字段类型并构建条目
                try:
                    sat_id = int(row[0])  # 卫星ID作为键
                    # 提取其他字段作为值字典
                    sat_data = {
                        'Bandwidth(MHz)': int(row[1]),
                        'SCS(kHz)': int(row[2]),
                        'RBs': int(row[3])
                    }
                    sat_id_dict[sat_id] = sat_data  # 存入字典（重复ID会覆盖）
                except ValueError as e:
                    raise ValueError(f"第{row_num}行数据类型转换失败：{e}")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在，请检查路径：{file_path}")
    except Exception as e:
        raise Exception(f"处理文件时发生错误：{e}")
    
    print(f"成功读取并转换{file_path}，共包含{len(sat_id_dict)}个卫星条目（以ID为键）")
    return sat_id_dict
def throughput_satisfaction(T_a, T_a_min,  beta_T):
    """
    计算吞吐满意度
    :param T_a: 当前吞吐量
    :param T_a_min: 吞吐量最小值
    :param beta_T: 控制平滑度的参数(5-10)
    :return: 吞吐满意度值
    """
    numerator = 1
    denominator = 1 + np.exp(-beta_T * (T_a - T_a_min))
    return numerator / denominator

def delay_satisfaction(D_a,  D_a_max, beta_D):
    """
    计算时延满意度
    :param D_a: 当前时延
    :param D_a_max: 时延最大值
    :param beta_D: 控制平滑度的参数(5-10)
    :return: 时延满意度值
    """
    numerator = 1
    denominator = 1 + np.exp(beta_D * (D_a - D_a_max))
    return numerator / denominator

def generate_app_weight_csv(apps, output_file):
    """
    生成包含卫星ID及对应权重（wa、aT、aD）的CSV文件，确保所有卫星wa总和为1，单个卫星aT+aD=1
    
    参数:
        apps: app的ID列表，例如[3743, 1666, 1644, 1143]
        output_file (str): 输出CSV文件的路径及名称，例如"apps_weights.csv"
    
    逻辑说明:
        1. 生成随机数作为初始权重，归一化后使所有app的wa总和为1
        2. 对每个app随机生成aT（0-1浮点数），计算aD=1-aT，保证aT+aD=1
        3. 将app的ID、wa、aT、aD写入CSV文件1
    """
    # 生成各app的wa（确保总和为1）
    random_weights = [random.random() for _ in apps]
    total = sum(random_weights)
    wa_list = [w / total for w in random_weights]
    
    # 生成每个卫星的aT和aD（aT + aD = 1）
    app_data = []
    for app_id, wa in zip(apps, wa_list):
        aT = random.random()
        aD = 1 - aT
        app_data.append([app_id, wa, aT, aD])
    
    # 写入CSV文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['app_id', 'wa', 'aT', 'aD'])
        writer.writerows(app_data)
    
    print(f"已生成CSV文件：{output_file}")

def read_app_weight_csv(input_file):
    """
    读取app权重CSV文件，返回嵌套字典结构，外层键为app的ID，内层字典包含wa、aT、aD权重
    
    参数:
        input_file (str): 要读取的CSV文件路径，例如"apps_weights.csv"
    
    返回:
        dict: 嵌套字典，格式为{app_id: {'wa': 权重值, 'aT': 权重值, 'aD': 权重值}}
    """
    app_dict = {}
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            app_id = int(row['app_id'])
            app_dict[app_id] = {
                'wa': float(row['wa']),
                'aT': float(row['aT']),
                'aD': float(row['aD'])
            }
    return app_dict

def generate_app_flow_csv(app_count, output_file):
    """
    生成包含app序号和随机flow_type（字符串类型）的CSV文件
    
    参数:
        app_count (int): app的数量，决定CSV文件的行数（序号从1到app_count）
        output_file (str): 输出CSV文件的路径及名称（如"app_flow.csv"）
    """
    # 定义flow_type的可选值列表
    flow_types = ["Voice", "V2X", "Video", "VR/AR"]
    
    # 准备数据：第一列为app序号（1到app_count），第二列为随机选择的flow_type
    app_data = []
    for app_id in range(1, app_count + 1):
        flow_type = random.choice(flow_types)  # 从列表中随机选择一个flow_type
        app_data.append([app_id, flow_type])
    
    # 写入CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['app序号', 'flow_type'])
        # 写入所有app数据
        writer.writerows(app_data)
    
    print(f"已生成CSV文件：{output_file}，包含{app_count}个app的信息")

def read_app_flow_csv(input_file):
    """
    读取包含app序号和flow_type的CSV文件，返回字典结构
    
    参数:
        input_file (str): 要读取的CSV文件路径（如"app_flow_types.csv"）
    
    返回:
        dict: 键为app序号（int），值为flow_type（str），格式如 {1: "Voice", 2: "V2X", ...}
    """
    app_flow_dict = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # 按表头解析每行数据
        for row in reader:
            # 转换app序号为整数，flow_type直接取字符串
            app_id = int(row['app序号'])
            flow_type = row['flow_type']
            app_flow_dict[app_id] = flow_type
    
    return app_flow_dict

def create_sat_obs_space(sat_id:int, max_users:int, num_other_sats:int, app_low:np.array, app_high:np.array, rbs:int) -> gym.spaces.Dict:
    """
    创建卫星的观测空间
    参数:
        sat_id: 卫星ID
        max_users: 最大用户数
        other_sats: 其他卫星的ID字典
    返回:
        gym.spaces.Dict: 卫星的观测空间
    """
    # 当前卫星的SNR空间，拆分成10份（对应0.1s -> 0.01s的时间粒度）
    snr_spaces = {
            f"snr_{j}": gym.spaces.Box(
                low=0.0, 
                high=100.0,
                shape=(max_users, rbs), # 用户数量 × RBs
                dtype=np.float32
            )
            for j in range(10)
    }
    observation_space = gym.spaces.Dict({
        "base_features": gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(4,),  # 4维特征
            dtype=np.float32
        ),
        "ele_angles": gym.spaces.Box(
            low=0.0, 
            high=90.0, 
            shape=(max_users,),  # M个用户
            dtype=np.float32
        ),
        "app_features": gym.spaces.Box(
            low=app_low, 
            high=app_high, 
            shape=(max_users, 7),  # M个app × 7维特征
            dtype=np.float32
        ),
        "other_satellites_snr": gym.spaces.Box(
            low=0.0, 
            high=100.0,
            shape=(num_other_sats, rbs),  # p颗星 × rbs数量
            dtype=np.float32
        ),
        **snr_spaces,
        "other_satellites_base": gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(num_other_sats, 2),  # p颗星 × 2维特征 (该星剩余RB比例，这颗星的业务满足率)
            dtype=np.float32
        ),
        "other_sat_ele_angle": gym.spaces.Box(
            low=0.0, 
            high=90.0, 
            shape=(num_other_sats, max_users),  # p颗星 × 用户数量特征
            dtype=np.float32
        ),
        "slices": gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(2,4),  # 2*4维特征
            dtype=np.float32
        )
    })
    return observation_space

