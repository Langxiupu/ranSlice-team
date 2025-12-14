import random
import numpy as np
import gym
from typing import Optional
import tool
from commModule.UserTerminal import Position, UserTerminal
from commModule.UserTerminal import Distance
from commModule.leoSat import LEOSatellite
from commModule.mac_scheduler import MacScheduler
from commModule.plot_channelSNR import TimeVaryingSatelliteChannel

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

class LEORANSlicing(gym.Env):
    def __init__(self, env_config):
        #* create users object
        #* create satellite object
        #* create channel object (remains to be surveyed)
        #* create slicing object
        #* create apps object
        #* create mac scheduler object
        # 首先创建卫星对象
        self.training = True
        self.lambda_param = env_config["lambda"]  # 读取lambda参数
        self.visibles_t_u = env_config["visibles_t_u"]  # visibles_t_u的文件路径
        self.common_sats = env_config["common_sats"]  # common_satellites是一个列表，包含卫星ID
        self.sats_bd_rbs = env_config["sats_bd_rbs"]  # 读取卫星带宽信息的文件路径
        self.apps_weights = env_config["apps_weights"]  # 读取卫星权重的文件路径
        self.app_flow_types = env_config["app_flow_types"]  # 读取应用流类型的文件路径
        self.leo_sats = [LEOSatellite(sat_id) for sat_id in self.common_sats]  # 创建LEOSatellite对象的列表
        sats_data = tool.read_sats_bd_rb_csv(self.sats_bd_rbs) # 读取卫星带宽信息
        self.apps_weights_dict = tool.read_app_weight_csv(self.apps_weights)  # 读取卫星权重信息
        for sat in self.leo_sats:  # 遍历所有卫星
            if sat.ID in sats_data:  # 如果卫星ID在卫星带宽信息中
                sat.config_BW(sats_data[sat.ID]['Bandwidth(MHz)'])  # 配置卫星带宽
                sat.config_rbs(sats_data[sat.ID]['RBs'])  # 配置卫星rbs数量
                sat.config_scs(sats_data[sat.ID]['SCS(kHz)'])  # 配置卫星子载波间隔
            else:  # 如果卫星ID不在卫星带宽信息中
                raise ValueError(f"卫星ID {sat.ID} 未找到对应的带宽信息")
        # 然后创建用户对象
        ue_position_dict = tool.read_positions_from_csv(env_config["ue_position"])  # 读取用户位置信息
        # 创建用户终端
        self.UEs = {}
        self.available_UEs = {}
        self.inavailable_UEs = {}
        self.ue_num = len(ue_position_dict)  # 用户数量
        # 读取流生成方式
        self.short_slot_interval = env_config["short_slot_interval"]  # 读取短时隙间隔
        self.slice_interval = env_config["slice_interval"]  # 读取切片间隔
        self.flow_gen_type = env_config["flow_gen_method"]["type"]  # 读取流生成方式的类型
        self.flow_gen_params = env_config["flow_gen_method"]["params"]  # 读取流生成方式的参数
        self.slice_slots = env_config["slice_slots"]  # 读取切片时隙数量
        self.current_slot = 0
        if self.flow_gen_type == "on-off":  # 如果流生成方式是"on-off"
            self.flow_gen_params = env_config["flow_gen_method"]["params"]  # 读取流生成方式的参数
        app_flow_types_dict = tool.read_app_flow_csv(self.app_flow_types)  # 读取应用流类型
        for ue_id, position in ue_position_dict.items():  # 遍历用户ID和位置信息
            flow_type = app_flow_types_dict[ue_id]  # 读取用户流类型
            user_terminal = UserTerminal(pos=position, common_satellites=self.common_sats, flow_type=flow_type, id=ue_id)  # 创建用户终端对象
            user_terminal.config_flow_method(self.flow_gen_type)  # 配置流生成方式
            user_terminal.config_flow_params(self.flow_gen_params)  # 配置流生成方式的参数
            self.UEs[ue_id] = user_terminal  # 将用户终端对象添加到用户字典中
            self.available_UEs[ue_id] = user_terminal  # 初始状态下所有用户终端均可用

        self.app_min_delay_demand =  0 # 单位 ms
        self.app_max_delay_demand =  200 # 单位 ms
        self.app_min_bandwidth_demand = 0 # 单位 Mbps
        self.app_max_bandwidth_demand = 15 # 单位 Mbps
        # 配置MacSheduler对象
        self.mac_scheduler = MacScheduler(policy="ADMM") # 当前mac调度策略为ADMM调度
        # 为每一个卫星创建对应的channelModel对象,每一个对象内部是每个ue组成的字典
        self.sats_channels = {} # 包含每个卫星对应的每个ue在0.1s粒度下的信道信息
        for sat in self.leo_sats:
            sat_channels = {}
            for ue_id, ue in self.UEs.items():
                sat_channel = TimeVaryingSatelliteChannel(total_time = self.slice_slots * 0.1, traj_time_step=0.1, trajectory_file=self.visibles_t_u, sat_id=sat.ID, usr_id=ue_id-1)
                sat_channels[ue_id] = sat_channel
            self.sats_channels[sat.ID] = sat_channels
        # 构建观测空间
        self.agent_ids = [str(sat.ID) for sat in self.leo_sats]
        app_low_item = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 每个APP的观测空间下界
        app_high_item = np.array([1000.0, 1.0, 1.0, 1.0, 1.0, 1000.0, 1000.0])  # 每个APP的观测空间上界
        app_low = np.tile(app_low_item, (self.ue_num, 1))  # 所有APP的观测空间下界
        app_high = np.tile(app_high_item, (self.ue_num, 1))
        self._observation_space = {
            str(sat.ID): tool.create_sat_obs_space(sat.ID, len(self.UEs), len(self.leo_sats)-1, app_low, app_high, sat.rbs)
            for sat in self.leo_sats
        }
        self.head_infos = []
        self.autoregressive_maps = []
        self.available = tool.calc_ratio_vec() # 得到的是一个时间跨度为3300秒，每0.1s记录当前地区用户活跃度的vector数组
        #首先是两种类型切片的带宽份额
        head_idx = 0
        self._actions = []
        for sat in self.leo_sats:
            # 首先是带宽密集型切片
            head_info_item = {}
            head_info_item["name"] = f"embb_{sat.ID}_bd"
            head_info_item["type"] = "categorical"
            head_info_item["out_dim"] = sat.rbs
            self.head_infos.append(head_info_item)
            autoregressive_map_item = []
            autoregressive_map_item.append(-1)
            self.autoregressive_maps.append(autoregressive_map_item)
            self._actions.append(gym.spaces.Discrete(sat.rbs))
            # 其次是延迟敏感性切片
            head_info_item = {}
            head_info_item["name"] = f"urllc_{sat.ID}_bd"
            head_info_item["type"] = "categorical"
            head_info_item["out_dim"] = sat.rbs
            self.head_infos.append(head_info_item)
            autoregressive_map_item = []
            autoregressive_map_item.append(head_idx)
            self.autoregressive_maps.append(autoregressive_map_item)
            self._actions.append(gym.spaces.Discrete(sat.rbs))
            head_idx += 2
        # 其次是两种切片的功率份额
        head_info_item = {}
        head_info_item["name"] = "embb_power"
        head_info_item["type"] = "continuous"
        head_info_item["out_dim"] = len(self.leo_sats)
        head_info_item["low"] = 0
        head_info_item["high"] = 1
        self.head_infos.append(head_info_item)
        self.autoregressive_maps.append([-1])
        self._actions.append(gym.spaces.Box(low=0, high=1, shape=(len(self.leo_sats),), dtype=np.float32))
        head_info_item = {}
        head_info_item["name"] = "urllc_power"
        head_info_item["type"] = "continuous"
        head_info_item["out_dim"] = len(self.leo_sats)
        head_info_item["low"] = 0
        head_info_item["high"] = 1
        self.head_infos.append(head_info_item)
        self.autoregressive_maps.append([-1])
        self._actions.append(gym.spaces.Box(low=0, high=1, shape=(len(self.leo_sats),), dtype=np.float32))
        # 每个app的目标卫星,是一个离散的动作，大小取决于公共可见卫星的数量
        for ue_id, ue in self.UEs.items():
            head_info_item = {}
            head_info_item["name"] = f"ue_{ue_id}_target_satellite"
            head_info_item["type"] = "categorical"
            head_info_item["out_dim"] = len(self.common_sats)
            self.head_infos.append(head_info_item)
            autoregressive_map_item = []
            autoregressive_map_item.append(-1)
            self.autoregressive_maps.append(autoregressive_map_item)
            self._actions.append(gym.spaces.Discrete(len(self.common_sats)))
        self._action_space = gym.spaces.Tuple(self._actions)
        observation = {}
        for agent_id in self.agent_ids:
            for field_name, field_value in self._observation_space[agent_id].items():
                # 生成带前缀的键（如sat_id=1，field_name=base_features→"1_base_features"）
                prefixed_key = f"{agent_id}_{field_name}"
                observation[prefixed_key] = field_value
        self.observation_space = gym.spaces.Dict(observation)
        


    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        #* install mac scheduler into satellites
        #* associate users' apps with slices
        #* generate the obs vector
        # 目前我们假定只有一个MacScheduler对象，因此直接取第一个，每个卫星都配置这个MacScheduler对象
        self.mac_scheduler = MacScheduler(policy="ADMM")
        distance_dict = tool.get_info_by_time(self.visibles_t_u, 0, self.common_sats, self.ue_num)  # 获取第0个时隙的用户-卫星距离信息
        # 其次需要对每一个ue进行重置
        for ue_id, ue in self.UEs.items():
            ue.reset()
        self.available_UEs = {}
        self.inavailable_UEs = {}
        # 将UEs中的ue初始时全部加入available_UEs
        for ue_id, ue in self.UEs.items():
            self.available_UEs[ue_id] = ue
        observation = {}
        sat_app_elevation_angles = {}
        for agent_id in self.agent_ids:
            sat_id = int(agent_id)
            app_elevation_angles = []
            for ue_id, ue in self.UEs.items():
                elevation_angle = distance_dict[ue_id][sat_id].altitude
                app_elevation_angles.append(elevation_angle)
            sat_app_elevation_angles[sat_id] = app_elevation_angles
        # 重置的话需要将observation_space设置为slot为0时的状态
        for agent_id in self.agent_ids:
            # 根据agent_id获取对应的卫星ID
            sat_id = int(agent_id)
            # 首先是四个基础特征 ：当前窗口进度，业务强度，RB比例，功率比例
            window_progress = 0.0
            traffic_intensity = self.available[0]
            rb_ratio = 1.0
            power_ratio = 1.0
            self._observation_space[agent_id]['base_features'] = np.array([
                window_progress,
                traffic_intensity,
                rb_ratio,
                power_ratio
            ], dtype=np.float32)
            # 其次是当前每个app相对当前卫星的仰角
            self._observation_space[agent_id]['ele_angles'] = np.array(sat_app_elevation_angles[sat_id], dtype=np.float32)
            # 接着是每个app的7个基本特征，当前队列长度，到达率估计，时延/带宽满意度，该app分配的带宽比例，带宽/时延需求
            app_features = []
            current_queue_lengths = []
            arrival_rate_estimates = []
            delay_satisfaction_ratios = []
            bandwidth_satisfaction_ratios = []
            allocated_bandwidth_ratios = []
            bandwidth_requirements = []
            delay_requirements = []
            for ue_id, ue in self.UEs.items():
                # 初始状态均为0
                current_queue_length = 0.0
                arrival_rate_estimate = 0.0
                delay_satisfaction_ratio = 0.0
                bandwidth_satisfaction_ratio = 0.0
                allocated_bandwidth_ratio = 0.0
                # 获取每个ue中绑定的app的qos需求
                app_delay_demand = ue.app.Delay_u
                app_bw_demand = ue.app.BW_u
                # 分别归一化
                delay_requirement = (self.app_max_delay_demand - app_delay_demand) / (self.app_max_delay_demand - self.app_min_delay_demand)
                bandwidth_requirement = (app_bw_demand - self.app_min_bandwidth_demand) / (self.app_max_bandwidth_demand - self.app_min_bandwidth_demand)
                current_queue_lengths.append(current_queue_length)
                arrival_rate_estimates.append(arrival_rate_estimate)
                delay_satisfaction_ratios.append(delay_satisfaction_ratio)
                bandwidth_satisfaction_ratios.append(bandwidth_satisfaction_ratio)
                allocated_bandwidth_ratios.append(allocated_bandwidth_ratio)
                bandwidth_requirements.append(bandwidth_requirement)
                delay_requirements.append(delay_requirement)
            app_features = np.array([
                current_queue_lengths,
                arrival_rate_estimates,
                delay_satisfaction_ratios,
                bandwidth_satisfaction_ratios,
                allocated_bandwidth_ratios,
                bandwidth_requirements,
                delay_requirements
            ], dtype=np.float32).T  # 转置为M×7的形状
            self._observation_space[agent_id]['app_features'] = app_features
            # 接下来包括对其他卫星的观测信息，包含四类，本区用户的平均SNR,剩余RB比例，业务满足率,其余卫星相对app的仰角，前者为一项，中间两者合并为一项，最后为一项
            other_sats_snr = []
            other_sats_base = []
            other_sats_ele_angles = []
            for other_sat in self.leo_sats:
                snr_total = []
                if other_sat.ID == sat_id:
                    continue
                # 计算本区用户的平均SNR
                for _ in range(other_sat.rbs):
                    snr_total.append(0)
                other_sats_snr.append(snr_total)
                # 剩余RB比例
                remaining_rb_ratio = 1.0
                # 该颗星的业务满足率
                satisfaction_ratio = 1.0
                other_sats_base.append([remaining_rb_ratio, satisfaction_ratio])
                other_sats_ele_angles.append(sat_app_elevation_angles[other_sat.ID])
            self._observation_space[agent_id]['other_satellites_snr'] = np.array(other_sats_snr, dtype=np.float32)
            self._observation_space[agent_id]['other_satellites_base'] = np.array(other_sats_base, dtype=np.float32)
            self._observation_space[agent_id]['other_sat_ele_angle'] = np.array(other_sats_ele_angles, dtype=np.float32)
            # 然后是当前卫星的snr空间，拆分成10份（对应0.1s -> 0.01s的时间粒度）
            snr_spaces = {}
            sat_rbs = 0
            for sat in self.leo_sats:
                if sat.ID != sat_id:
                    continue
                sat_rbs = sat.rbs
            # 初始化信道
            for ue_id, ue in self.UEs.items():
                sat_ue_channel = self.sats_channels[other_sat.ID][ue_id]
                sat_ue_channel.bandwidth = other_sat._BW * 1e6  # 转换为Hz
                sat_ue_channel.scs = other_sat._scs * 1e3  # 转换为Hz
                sat_ue_channel.total_subcarriers = sat_ue_channel.num_rb * sat_ue_channel.num_sc_per_rb
                sat_ue_channel._generate_static_parameters()
                sat_ue_channel.initialize_channel()
            for j in range(10):
                if f"snr_{j}" not in snr_spaces:
                    # 用用户数 * RB数来初始化
                    snr_spaces[f"snr_{j}"] = []
                    for ue_id, ue in self.UEs.items():
                        snr_space = []
                        for _ in range(sat_rbs):
                            snr_space.append(0.0)
                        snr_spaces[f"snr_{j}"].append(snr_space)
                    self._observation_space[agent_id][f"snr_{j}"] = np.array(snr_spaces[f"snr_{j}"], dtype=np.float32)
            # 最后是切片特征，第一个是embb切片的特征，第二个是urllc切片的特征
            # embb切片特征
            embb_slice_feature = [0.0, 0.0, 0.0, 0.0] 
            # urllc切片特征
            urllc_slice_feature = [0.0, 0.0, 0.0, 0.0] 
            self._observation_space[agent_id]['slices'] = np.array([embb_slice_feature, urllc_slice_feature], dtype=np.float32)
        # 核心修改：将当前agent的所有字段按"前缀_字段名"存入observation
        # 遍历self.observation_space[agent_id]的所有字段
            for field_name, field_value in self._observation_space[agent_id].items():
                # 生成带前缀的键（如sat_id=1，field_name=base_features→"1_base_features"）
                prefixed_key = f"{sat_id}_{field_name}"
                observation[prefixed_key] = field_value
        self.observation_space = gym.spaces.Dict(observation)
        self.current_slot = 0
        # 给observation添加两个字段is_first和is_terminal
        observation["is_first"] = True
        observation["is_terminal"] = False
        return observation
    
    def step(self, actions):
        #* schedule RBs for each slice 
        #* repeat the above process for all satellites
        # 我们首先需要拆解actions中的所有动作并分类，后续再做处理
        info = {}
        action_dict = {}
        # 遍历common_sats中的卫星ID,在action_dict中创建对应的字典
        for sat_id in self.common_sats:
            action_dict[sat_id] = {}
        for i, ac in enumerate(actions):
            head_name = self.head_infos[i]["name"]
            if "embb_" in head_name and "_bd" in head_name:
                # 代表这是某颗卫星的带宽密集型切片的RB分配动作
                # 从该动作中选择参数最大的那个RB作为分配结果
                sat_id = int(head_name.split("_")[1])
                action = np.argmax(ac)
                action_dict[sat_id]["embb_bd"] = action
            elif "urllc_" in head_name and "_bd" in head_name:
                # 代表这是某颗卫星的延迟敏感型切片的RB分配动作
                sat_id = int(head_name.split("_")[1])
                action = np.argmax(ac)
                action_dict[sat_id]["urllc_bd"] = action
            elif "embb_power" in head_name: 
                # 代表这是某颗卫星的带宽密集型切片的功率分配动作
                # 遍历ac的每个元素，根据下标去common_sats中找对应的卫星ID
                for j, power_ratio in enumerate(ac):
                    sat_id = self.common_sats[j]
                    action_dict[sat_id]["embb_power"] = power_ratio
            elif "urllc_power" in head_name:
                # 代表这是某颗卫星的延迟敏感型切片的功率分配动作
                for j, power_ratio in enumerate(ac):
                    sat_id = self.common_sats[j]
                    action_dict[sat_id]["urllc_power"] = power_ratio
            elif "ue_" in head_name and "_target_satellite" in head_name:
                # 代表这是某个用户终端选择目标卫星的动作
                ue_id = head_name.split("_")[1]
                action = np.argmax(ac)
                target_sat_id = self.common_sats[action]
                action_dict[target_sat_id].setdefault("ue_targets", []).append(ue_id)
        # 首先我们需要获取当前所有处在活跃状态的用户终端
        config_terminals(self.available_UEs, self.inavailable_UEs, int(len(self.UEs) * self.available[self.current_slot]))
        # 然后我们需要根据action_dict中的内容，对每颗卫星进行资源分配
        # 我们先遍历可用的用户终端，将他们连接到对应的卫星上
        for ue_id, ue in self.available_UEs.items():
            # 首先找到该用户终端选择的目标卫星
            target_sat_id = None
            for sat_id, actions in action_dict.items():
                if "ue_targets" in actions and ue_id in actions["ue_targets"]:
                    target_sat_id = sat_id
                    break
            if target_sat_id is not None:
                # 找到目标卫星，进行连接, 先判断与原来连接的卫星是否相同
                if ue.connected_satellite.ID != target_sat_id:
                    # 如果不同，先断开原来的连接
                    ue.disconnect_sat()
                    # 然后连接到新的卫星
                    for sat in self.leo_sats:
                        if sat.ID == target_sat_id:
                            ue.connect_specific_sat(sat)
                            break
        sat_snr_dict = {}
        sat_snr_dict_real = {}
        # 如果是训练模式
        if self.training == True:
            # 我们需要根据sats_channels_details得到每个卫星每个对应时隙下对应每个ue的snr信息
            for sat in self.leo_sats:
                sat_snr_dict[sat.ID] = {}
                sat_snr_dict_real[sat.ID] = {}
                for ue_id, ue in self.UEs.items():
                    for j in range(10):
                        sat_ue_channel = self.sats_channels[sat.ID][ue_id]
                        snr = sat_ue_channel.get_rb_cnr()
                        sat_snr_dict[sat.ID][ue_id][j] = snr
                        sat_snr_dict_real[sat.ID][ue_id][j] = snr
                        sat_ue_channel.step_channel()
        # mac_scheduler需要接收的参数有leo_sats, action_dict, UEs, sat_snr_dict
        # 返回的参数包括
        # 对应app是否满足要求，只有带宽和时延两个维度均满足才认为满足要求，返回一个dict, key为app_id, value为一个dict，包含了当前app的连接卫星ID,切片类型,满意度的值
        app_results, slice_results = self.mac_scheduler.statistics_summary(self.leo_sats, action_dict, self.UEs, sat_snr_dict)
        k = 1 # 该参数用来控制功率相对带宽的价格比例
        S_delay = {}  # 每个app的时延满意度
        S_throughput = {}  # 每个app的带宽满意度
        # 尝试遍历app_results中的每个app，填充S_delay和S_throughput
        for app_id, app_result in app_results.items():
            S_delay[app_id] = app_result["delay_satisfaction"]
            S_throughput[app_id] = app_result["throughput_satisfaction"]
        # 根据app_results, slice_results统计每颗卫星的业务满足率以及每颗卫星下两种切片的到达率
        sat_results = {}
        for sat in self.leo_sats:
            sat_id = sat.ID
            sat_results[sat_id] = {}
            # 先遍历app_results中的每个app,统计所有在当前卫星下满足qos要求的app
            total_app_num = 0
            satisfied_app_num = 0
            sat_results[sat_id]["embb"] = {}
            sat_results[sat_id]["embb"]["arrival_rate"] = 0.0
            sat_results[sat_id]["embb"]["poorest_delay_satisfaction"] = 1.0
            sat_results[sat_id]["embb"]["poorest_throughput_satisfaction"] = 1.0
            sat_results[sat_id]["urllc"] = {}
            sat_results[sat_id]["urllc"]["arrival_rate"] = 0.0
            sat_results[sat_id]["urllc"]["poorest_delay_satisfaction"] = 1.0
            sat_results[sat_id]["urllc"]["poorest_throughput_satisfaction"] = 1.0
            for app_id, app_result in app_results.items():
                if app_result["satellite_id"] == sat_id:
                    total_app_num += 1
                    if app_result['qos_met'] == True:
                        satisfied_app_num += 1
                    if app_result["slice_type"] == "eMBB":
                        sat_results[sat_id]["embb"]["arrival_rate"] += (app_result['avg_transmit_rate']+0.0) / 1e6
                        if app_result["delay_satisfaction"] < sat_results[sat_id]["embb"]["poorest_delay_satisfaction"]:
                            sat_results[sat_id]["embb"]["poorest_delay_satisfaction"] = app_result["delay_satisfaction"]
                        if app_result["throughput_satisfaction"] < sat_results[sat_id]["embb"]["poorest_throughput_satisfaction"]:
                            sat_results[sat_id]["embb"]["poorest_throughput_satisfaction"] = app_result["throughput_satisfaction"]
                    elif app_result["slice_type"] == "uRLLC":
                        sat_results[sat_id]["urllc"]["arrival_rate"] += (app_result['avg_transmit_rate']+0.0) / 1e6
                        if app_result["delay_satisfaction"] < sat_results[sat_id]["urllc"]["poorest_delay_satisfaction"]:
                            sat_results[sat_id]["urllc"]["poorest_delay_satisfaction"] = app_result["delay_satisfaction"]
                        if app_result["throughput_satisfaction"] < sat_results[sat_id]["urllc"]["poorest_throughput_satisfaction"]:
                            sat_results[sat_id]["urllc"]["poorest_throughput_satisfaction"] = app_result["throughput_satisfaction"]
            # 计算业务满足率
            if total_app_num == 0:
                sat_results[sat_id]["app_satisfaction"] = 1.0
            else:
                sat_results[sat_id]["app_satisfaction"] = (satisfied_app_num+0.0) / total_app_num
            # 对每个切片的业务到达率归一化，指标用总用户数量 * 最大带宽要求
            sat_results[sat_id]["embb"]["arrival_rate"] = sat_results[sat_id]["embb"]["arrival_rate"] / (self.ue_num * self.app_max_bandwidth_demand)
            sat_results[sat_id]["urllc"]["arrival_rate"] = sat_results[sat_id]["urllc"]["arrival_rate"] / (self.ue_num * self.app_max_bandwidth_demand)
        # 计算reward
        # 关于我们reward的计算，有两部分组成
        # 第一部分是系统满意度
        system_satisfaction = 0.0
        for ue_id, ue in self.UEs.items():
            system_satisfaction += self.apps_weights_dict[ue_id]["wa"] * (self.apps_weights_dict[ue_id]["aT"] * S_throughput[ue_id] + self.apps_weights_dict[ue_id]["aD"] * S_delay[ue_id])
        # 第二部分是资源使用成本
        resource_cost = 0.0
        for sat in self.leo_sats:
            sat_id = sat.ID
            sat_bw_max = sat.BW
            sat_power_max = sat.p_max
            embb_bd = action_dict[sat_id]["embb_bd"]
            urllc_bd = action_dict[sat_id]["urllc_bd"]
            embb_bd_ratio = (embb_bd + 0.0) / sat.rbs
            urllc_bd_ratio = (urllc_bd + 0.0) / sat.rbs
            embb_power_ratio = action_dict[sat_id]["embb_power"]
            urllc_power_ratio = action_dict[sat_id]["urllc_power"]
            if embb_power_ratio + urllc_power_ratio > 1.0: # 如果功率比例之和超过1，则进行归一化处理
                total_power = embb_power_ratio + urllc_power_ratio
                embb_power_ratio = embb_power_ratio / total_power
                urllc_power_ratio = urllc_power_ratio / total_power
            # 计算该卫星的资源使用成本
            resource_cost += self.lambda_param * (embb_bd_ratio * sat_bw_max + urllc_bd_ratio * sat_bw_max + k * (embb_power_ratio * sat_power_max + urllc_power_ratio * sat_power_max))
        reward = system_satisfaction - resource_cost
        # 接下来我们需要生成该时隙的observation
        observation = {}
        distance_dict = tool.get_info_by_time(self.visibles_t_u, self.current_slot, self.common_sats, self.ue_num)
        sat_app_elevation_angles = {}
        for agent_id in self.agent_ids:
            sat_id = int(agent_id)
            app_elevation_angles = []
            for ue_id, ue in self.UEs.items():
                elevation_angle = distance_dict[ue_id][sat_id].altitude
                app_elevation_angles.append(elevation_angle)
            sat_app_elevation_angles[sat_id] = app_elevation_angles
        for sat in self.leo_sats:
            sat_id = sat.ID
            # 首先是四个基础特征 ：当前窗口进度，业务强度，RB比例，功率比例
            window_progress = (self.current_slot + 0.0) / (self.slice_slots-1)
            traffic_intensity = len(self.available_UEs) / self.ue_num
            rb_ratio = 1 - (action_dict[sat_id]["embb_bd"] + action_dict[sat_id]["urllc_bd"] + 0.0) / sat.rbs
            if action_dict[sat_id]["embb_power"] + action_dict[sat_id]["urllc_power"] > 1.0:
                power_ratio = 0.0
            else:
                power_ratio = 1 - (action_dict[sat_id]["embb_power"] + action_dict[sat_id]["urllc_power"] + 0.0)
            self._observation_space[str(sat_id)] = np.array([window_progress, traffic_intensity, rb_ratio, power_ratio], dtype=np.float32)
            # 其次是当前每个app相对当前卫星的仰角
            self._observation_space[str(sat_id)]['app_elevation_angles'] = np.array(sat_app_elevation_angles[sat_id], dtype=np.float32)
            # 接着是每个app的7个基本特征，当前队列长度，到达率估计，时延/带宽满意度，该app分配的带宽比例，带宽/时延需求
            app_features = []
            current_queue_lengths = []
            arrival_rate_estimates = []
            delay_satisfaction_ratios = []
            bandwidth_satisfaction_ratios = []
            allocated_bandwidth_ratios = []
            bandwidth_requirements = []
            delay_requirements = []
            for ue_id, ue in self.UEs.items():
                current_queue_length = ue.queue_length()
                arrival_rate_estimate = 0.0
                arrival_rate_estimate = ((app_results[ue_id]['avg_transmit_rate']+0.0)/1e6 - self.app_min_bandwidth_demand) / (self.app_max_bandwidth_demand - self.app_min_bandwidth_demand)
                sat_id = ue.satellite.ID
                sat = self.leo_sats[sat_id]
                sat_id = sat.ID
                sat = self.leo_sats[sat_id]
                sat_id = sat.ID
                sat = self.leo_sats[sat_id]
                delay_satisfaction_ratio = S_delay[ue_id]
                bandwidth_satisfaction_ratio = S_throughput[ue_id]
                allocated_bandwidth_ratio = (action_dict[sat_id]["embb_bd"]+0.0) / sat.rbs
                # 获取每个ue中绑定的app的qos需求
                app_delay_demand = ue.app.Delay_u
                app_bw_demand = ue.app.BW_u
                # 分别归一化
                delay_requirement = (self.app_max_delay_demand - app_delay_demand) / (self.app_max_delay_demand - self.app_min_delay_demand)
                bandwidth_requirement = (app_bw_demand - self.app_min_bandwidth_demand) / (self.app_max_bandwidth_demand - self.app_min_bandwidth_demand)
                current_queue_lengths.append(current_queue_length)
                arrival_rate_estimates.append(arrival_rate_estimate)
                delay_satisfaction_ratios.append(delay_satisfaction_ratio)
                bandwidth_satisfaction_ratios.append(bandwidth_satisfaction_ratio)
                allocated_bandwidth_ratios.append(allocated_bandwidth_ratio)
                bandwidth_requirements.append(bandwidth_requirement)
                delay_requirements.append(delay_requirement)
            app_features = np.array([current_queue_lengths, arrival_rate_estimates, delay_satisfaction_ratios, bandwidth_satisfaction_ratios, allocated_bandwidth_ratios, bandwidth_requirements, delay_requirements], dtype=np.float32).T
            self._observation_space[str(sat_id)]['app_features'] = app_features
            # 接下来包括对其他卫星的观测信息，包含四类，本区用户的平均SNR,剩余RB比例，业务满足率,其余卫星相对app的仰角，前者为一项，中间两者合并为一项，最后为一项
            other_sats_snr = []
            other_sats_base = []
            other_sats_ele_angles = []
            for other_sat in self.leo_sats:
                snr_total = np.zeros(other_sat.rbs, dtype=np.float32)
                if other_sat.ID == sat_id:
                    continue
                for ue_id, ue in self.UEs.items():
                    for j in range(10):
                        snr = (sat_snr_dict_real[other_sat.ID][ue_id][j])/10/len(self.UEs)
                        snr_total += snr
                other_sats_snr.append(snr_total.tolist())
                # 剩余RB比例
                remaining_rb_ratio = 1.0 - (action_dict[other_sat.ID]["embb_bd"] + action_dict[other_sat.ID]["urllc_bd"] + 0.0) / other_sat.rbs
                # 该颗星的业务满足率
                satisfaction_ratio = sat_results[sat_id]["app_satisfaction"]
                other_sats_base.append([remaining_rb_ratio, satisfaction_ratio])
                other_sats_ele_angles.append(sat_app_elevation_angles[other_sat.ID])
            self._observation_space[str(sat_id)]['other_satellites_snr'] = np.array(other_sats_snr, dtype=np.float32)
            self._observation_space[str(sat_id)]['other_satellites_base'] = np.array(other_sats_base, dtype=np.float32)
            self._observation_space[str(sat_id)]['other_sat_ele_angle'] = np.array(other_sats_ele_angles, dtype=np.float32)
            # 然后是当前卫星的snr空间，拆分成10份（对应0.1s -> 0.01s的时间粒度）
            snr_spaces = {}
            sat_snr_info = sat_snr_dict[sat.ID]
            for j in range(10):
                if f"snr_{j}" not in snr_spaces:
                    # 用用户数 * RB数来初始化
                    snr_spaces[f"snr_{j}"] = []
                    for ue_id, ue in self.UEs.items():
                        snr_space = sat_snr_info[ue_id][j].tolist()
                        snr_spaces[f"snr_{j}"].append(snr_space)
                    self._observation_space[str(sat_id)][f"snr_{j}"] = np.array(snr_spaces[f"snr_{j}"], dtype=np.float32)     
            # 最后是切片特征，第一个是embb切片的特征，第二个是urllc切片的特征
            # embb切片特征
            embb_slice_feature = [sat_results[sat_id]["embb"]["arrival_rate"],
                                   sat_results[sat_id]["embb"]["poorest_delay_satisfaction"], 
                                   sat_results[sat_id]["embb"]["poorest_throughput_satisfaction"], 
                                   (action_dict[sat_id]["embb_bd"]+0.0) / sat.rbs]
            # urllc切片特征
            urllc_slice_feature = [sat_results[sat_id]["urllc"]["arrival_rate"],
                                   sat_results[sat_id]["urllc"]["poorest_delay_satisfaction"], 
                                   sat_results[sat_id]["urllc"]["poorest_throughput_satisfaction"], 
                                   (action_dict[sat_id]["urllc_bd"]+0.0) / sat.rbs]
            self._observation_space[str(sat_id)]['slices'] = np.array([embb_slice_feature, urllc_slice_feature], dtype=np.float32)                   
            # 核心修改：将当前agent的所有字段按"前缀_字段名"存入observation
            # 遍历self.observation_space[agent_id]的所有字段
            for field_name, field_value in self._observation_space[str(sat_id)].items():
                # 生成带前缀的键（如sat_id=1，field_name=base_features→"1_base_features"）
                prefixed_key = f"{sat_id}_{field_name}"
                observation[prefixed_key] = field_value
        self.observation_space = gym.spaces.Dict(observation)
        self.current_slot += 1
        done = (self.current_slot == self.slice_slots)
        observation["is_first"] = False 
        observation["is_terminal"] = done
        return observation, reward, done, info

    def seed(self, seed):
        # 保持不变：设置随机种子
        np.random.seed(seed)

    @property
    def action_space(self):
        # 保持不变：动作空间属性
        return self._action_space
    
    @property
    def action_head_infos(self):
        return self.head_infos
    
    @property
    def action_autoregressive_maps(self):
        return self.autoregressive_maps
