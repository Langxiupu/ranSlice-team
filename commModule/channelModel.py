import numpy as np
import pandas as pd
from scipy.stats import rice, rayleigh
from scipy.special import j0  # 引入贝塞尔函数

import matplotlib.pyplot as plt
import pickle

from torch import Value


class TimeVaryingSatelliteChannel:
    def __init__(self, num_time_slots=100, time_step=1e-3, trajectory_file=None, user_id=None, sat_id=None):
        # 系统参数
        self.bandwidth = 20e6
        self.scs = 15e3
        self.num_rb = 106
        self.num_sc_per_rb = 12
        self.total_subcarriers = self.num_rb * self.num_sc_per_rb
        if user_id is None:
            raise ValueError("user_id is None")
        self.user_id = user_id
        if sat_id is None:
            raise ValueError("sat_id is None")
        self.sat_id = sat_id

        # 卫星通信参数
        self.carrier_freq = 2e9
        self.tx_power = 30  # 30 dBm
        self.tx_antenna_gain = 20
        self.rx_antenna_gain = 10
        self.noise_figure = 4.1
        
        self.distance = None
        self.elevation_angle = None
        
        # 轨迹文件
        self.trajectory_file = trajectory_file
        self.trajectory_df = None
        self.load_trajectory_data()
        
        # 卫星运动速度（假设已知）
        self.satellite_speed = 7.5e3  # 7.5 km/s 典型低轨卫星速度
        
        # 时间参数
        self.num_time_slots = num_time_slots
        self.time_step = time_step
        self.current_slot = -1
        
        # 多径参数
        self.num_taps = 3
        self.max_delay_spread = 250e-9
        self.b_ssr = 0.126
        self.omega = 0.835
        self.total_power_tap0 = self.omega + 2 * self.b_ssr
        
        # 预生成固定参数
        self._generate_static_parameters()
    
    def load_trajectory_data(self):
        """
        读取.pkl文件获取卫星的轨迹数据
        visibles_t_u-46users-filtered.pkl的格式
        visibles_t_u: list 长度为2000, 代表以0.1s为间隔的2000个时刻
        visibles_t_u[i]: dict 长度为46, 键为用户id [0,.., 45, ], 值代表46个用户相对于公共卫星的距离
        visibles_t_u[i][j]: dict 长度为4 key为可见的卫星ID, value为相对于可见卫星的一些距离属性。公共的可见卫星ID为[1143, 1644, 1666, 3743]
        visibles_t_u[i][j][1143]: VisibleSat对象, 定义见common.utils.VisibleSat 
        """
        if self.trajectory_file is None:
            raise FileNotFoundError

        with open (self.trajectory_file, 'rb') as f:
            visible_t_u = pickle.load(f)

        # 初始化第一组值
        self.elevation_angle = np.zeros((len(visible_t_u,)))
        self.distance = np.zeros((len(visible_t_u,)))

        for i in range(len(visible_t_u)):
            self.distance[i] = visible_t_u[i][self.user_id][self.sat_id].range
            self.elevation_angle[i] = visible_t_u[i][self.user_id][self.sat_id].altitude
    
    def update_satellite_parameters(self, slot_idx):
        """更新当前时隙的卫星位置参数"""
        self.elevation_angle = self.elevation_data[slot_idx]
        self.distance = self.distance_data[slot_idx]
        
        # 更新最大多普勒频移
        self._update_max_doppler()
        
        # 更新自由空间路径损耗
        self._update_fspl()
        
        # 更新多普勒频移配置
        self._update_doppler_shifts()
        
        # 更新时间相关系数（基于Jakes模型）
        self._update_time_correlation()
    
    def _update_max_doppler(self):
        """更新最大多普勒频移"""
        self.max_doppler = (self.satellite_speed * self.carrier_freq * 
                            np.cos(np.radians(self.elevation_angle))) / 3e8
    
    def _update_fspl(self):
        """更新自由空间路径损耗"""
        distance_km = self.distance / 1000
        carrier_freq_MHz = self.carrier_freq / 1e6
        self.fspl = -32.5 - 20 * np.log10(distance_km) - 20 * np.log10(carrier_freq_MHz)
    
    def _update_time_correlation(self):
        """更新时间相关系数"""
        self.rho = j0(2 * np.pi * abs(self.max_doppler) * self.time_step)
    
    def _update_doppler_shifts(self):
        """更新多普勒频移配置（考虑当前最大多普勒）"""
        # 生成多普勒频移（各径独立）
        shifts = np.zeros(self.num_taps)
        shifts[0] = self.max_doppler  # LOS路径
        
        # 散射路径用Jakes模型
        def jakes_doppler(max_dop, num_paths):
            angles = np.linspace(0, 2*np.pi, num_paths+1)[:-1]
            return max_dop * np.cos(angles)
        
        if self.num_taps > 1:
            shifts[1:] = jakes_doppler(self.max_doppler, self.num_taps-1)
        
        self.doppler_shifts = shifts
    
    def _generate_static_parameters(self):
        # 1. 计算基础参数（不变部分）
        self.freqs = np.arange(
            -self.bandwidth/2 + self.scs/2, 
            self.bandwidth/2, 
            self.scs
        )
        
        # 2. 生成固定多径结构
        self.tap_delays = np.zeros(self.num_taps)
        if self.num_taps > 1:
            self.tap_delays[1:] = np.sort(
                np.random.uniform(0, self.max_delay_spread, self.num_taps-1)
            )
        
        # 3. 计算各抽头功率分配
        decay_factor = 3.0 / self.max_delay_spread
        pdp = np.exp(-decay_factor * self.tap_delays)
        pdp[0] = self.total_power_tap0
        self.pdp_normalized = pdp / np.sum(pdp)
        
        # 4. 生成固定分量（LOS和散射基线）
        self.v0 = np.sqrt(self.omega) * np.exp(1j * np.random.uniform(0, 2*np.pi))
        self.sigma0 = np.sqrt(self.b_ssr)  # 散射分量标准差
        
        # 5. 初始多普勒频移配置
        self._update_doppler_shifts()
    
    def initialize_channel(self):
        # 更新卫星位置参数（第一个时隙）
        self.update_satellite_parameters(0)
        
        # 存储时变信道状态
        self.tap_scattering = np.zeros((self.num_taps, self.num_time_slots), dtype=complex)
        
        # 初始化首个时隙的散射分量
        # 抽头0：莱斯信道
        nlos0 = self.sigma0 * (np.random.randn() + 1j * np.random.randn())
        self.tap_scattering[0, 0] = nlos0
        
        # 其他抽头：瑞利信道
        for i in range(1, self.num_taps):
            scale = np.sqrt(self.pdp_normalized[i] / 2)
            self.tap_scattering[i, 0] = scale * (np.random.randn() + 1j * np.random.randn())
        
        self.current_slot = 0
    
    def step_channel(self):
        """更新到下一时隙的信道状态"""
        if self.current_slot >= self.num_time_slots - 1:
            raise ValueError("Reached end of simulation period")
        
        # 推进到时隙
        self.current_slot += 1
        
        # 更新卫星位置参数
        self.update_satellite_parameters(self.current_slot)
        
        # 更新每个抽头的散射分量（使用AR(1)模型）
        noise_scaling = np.sqrt(1 - self.rho**2)
        for i in range(self.num_taps):
            prev_val = self.tap_scattering[i, self.current_slot-1]
            
            # 生成新随机噪声（功率与抽头匹配）
            if i == 0:
                noise_scale = self.sigma0 * noise_scaling
            else:
                scale = np.sqrt(self.pdp_normalized[i] / 2)
                noise_scale = scale * noise_scaling
                
            noise = noise_scale * (np.random.randn() + 1j * np.random.randn())
            
            # AR(1) 更新
            self.tap_scattering[i, self.current_slot] = self.rho * prev_val + noise
    
    def get_current_channel(self):
        """获取当前时隙的信道响应"""
        if self.current_slot < 0:
            raise RuntimeError("Channel not initialized")
        
        # 组装当前抽头增益（LOS分量只在抽头0存在）
        tap_gains = np.zeros(self.num_taps, dtype=complex)
        tap_gains[0] = self.v0 + self.tap_scattering[0, self.current_slot]
        for i in range(1, self.num_taps):
            tap_gains[i] = self.tap_scattering[i, self.current_slot]
        
        # 应用发射功率
        tx_power_watts = 10**((self.tx_power - 30) / 10)
        tap_gains *= np.sqrt(tx_power_watts)
        
        # 计算频率响应（考虑多普勒）
        h_freq = np.zeros(len(self.freqs), dtype=complex)
        for i, f in enumerate(self.freqs):
            for k in range(self.num_taps):
                freq_shifted = f + self.doppler_shifts[k]
                phase_shift = 2 * np.pi * freq_shifted * self.tap_delays[k]
                h_freq[i] += tap_gains[k] * np.exp(-1j * phase_shift)
                
        return h_freq
    
    def calculate_rb_snr_for_slot(self):
        """计算当前时隙的RB级SNR"""
        h_freq = self.get_current_channel()
        channel_gain_db = 20 * np.log10(np.abs(h_freq))
        
        # 接收功率（每子载波）
        tx_power_per_sub = (self.tx_power - 
                           10 * np.log10(self.total_subcarriers))
        rx_power_per_sub = (tx_power_per_sub + self.tx_antenna_gain +
                           self.rx_antenna_gain + self.fspl + channel_gain_db)
        
        # 噪声功率（每子载波）
        kT = -174  # dBm/Hz
        noise_power_per_sub = kT + 10 * np.log10(self.scs) + self.noise_figure
        
        # 子载波SNR
        snr_per_sub = rx_power_per_sub - noise_power_per_sub
        
        # 转换为RB级SNR（取RB内12个子载波的平均）
        snr_per_rb = np.zeros(self.num_rb)
        for rb in range(self.num_rb):
            start_idx = rb * self.num_sc_per_rb
            end_idx = start_idx + self.num_sc_per_rb
            snr_per_rb[rb] = np.mean(snr_per_sub[start_idx:end_idx])
            
        return snr_per_rb


if __name__ == "__main__":
    # 1. 加载卫星的轨迹数据

    # 2. 创建信道模型
    channel = TimeVaryingSatelliteChannel(trajectory_file=...)
    channel.get_current_channel()
    snr_per_rb = channel.calculate_rb_snr_for_slot()

    # plot
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
    ax.bar(range(len(snr_per_rb)), snr_per_rb)
    ax.set_title('SNR per Resource Block')
    ax.set_xlabel('RB Index')
    ax.set_ylabel('SNR (dB)')
    ax.grid(True)

    plt.show()

