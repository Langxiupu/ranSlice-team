import numpy as np
import pickle
from scipy.special import j0
import matplotlib.pyplot as plt

SEED = 202510
np.random.seed(SEED)


class TimeVaryingSatelliteChannel:
    """
    时变多径卫星信道（频率选择性 + 时间相关），
    采用严格逐时隙（online）轨迹读取方式。

    - 轨迹粒度：traj_time_step（例如 0.1s）
    - 信道粒度：time_step（例如 0.01s）
    """

    def __init__(self,
                 total_time=200.0,           # 总仿真时间（秒）
                 time_step=0.01,            # 信道粒度（秒）
                 trajectory_file=None,
                 usr_id=0,
                 sat_id=0,
                 traj_time_step=0.1         # 轨迹分辨率（秒）
                 ):

        self.total_time = total_time
        self.time_step = time_step
        self.traj_time_step = traj_time_step
        self.num_time_slots = int(total_time / time_step)

        self.usr_id = usr_id
        self.sat_id = sat_id

        # 载波和系统参数
        self.bandwidth = 20e6
        self.scs = 15e3
        self.num_rb = 108
        self.num_sc_per_rb = 12
        self.total_subcarriers = self.num_rb * self.num_sc_per_rb

        self.carrier_freq = 2e9
        self.tx_antenna_gain = 10.0
        self.rx_antenna_gain = 10.0

        self.noise_psd_dbmhz = -204.0
        self.noise_figure_db = 0.0

        # 多径结构
        self.num_taps = 3
        self.max_delay_spread = 250e-9
        self.b_ssr = 0.126
        self.omega = 0.835
        self.total_power_tap0 = self.omega + 2 * self.b_ssr

        # 一些大尺度/状态变量先给默认值，避免初始化顺序问题
        self.elevation_angle = 0.0
        self.distance = 1e6          # 随便给个远距离
        self.max_doppler = 0.0
        self.fspl = 0.0
        self.rho = 0.0

        # 加载轨迹（但不展开）
        self._load_raw_trajectory(trajectory_file)

        # 时隙状态
        self.current_slot = -1
        self.visible_now = False  # 当前时隙是否可见

        # 多径固定参数（内部会用到 max_doppler，所以必须在它已经定义之后调用）
        self._generate_static_parameters()

    # ===============================================================
    # 轨迹相关（逐时隙加载）
    # ===============================================================

    def _load_raw_trajectory(self, trajectory_file):
        """读取 pkl，但不展开，只保存原始轨迹。"""
        if trajectory_file is None:
            raise FileNotFoundError("必须提供 trajectory_file")

        with open(trajectory_file, "rb") as f:
            self.traj = pickle.load(f)

        self.traj_len = len(self.traj)  # 例如 2000 帧（0.1s 粒度）

    def _get_traj_index(self, slot_idx):
        """将信道时隙 idx 映射到轨迹文件的索引。"""
        j = int((slot_idx * self.time_step) / self.traj_time_step)
        # 超过轨迹末尾 → 按不可见处理
        if j >= self.traj_len:
            return None
        return j

    def _update_satellite_parameters(self, slot_idx):
        """逐时隙更新几何参数；不可见时标记 visible_now=False。"""

        traj_idx = self._get_traj_index(slot_idx)
        if traj_idx is None:
            # 轨迹之外 → 不可见
            self.visible_now = False
            return

        user_dict = self.traj[traj_idx][self.usr_id]

        if self.sat_id not in user_dict:
            # 此时刻不可见
            self.visible_now = False
            return

        # 可见：更新 elevation 和 distance
        sat = user_dict[self.sat_id]
        self.elevation_angle = sat.altitude
        self.distance = sat.range
        self.visible_now = True

        # 更新大尺度参数
        self._update_max_doppler(traj_idx)
        self._update_fspl()
        self._update_time_correlation()

    # ===============================================================
    # 大尺度：多普勒、路径损耗、时间相关
    # ===============================================================

    def _update_max_doppler(self, traj_idx):
        """用轨迹层距离差推相对径向速度。"""
        c = 3e8

        if traj_idx == 0:
            self.max_doppler = 0.0
            return

        prev = self.traj[traj_idx - 1][self.usr_id]
        curr = self.traj[traj_idx][self.usr_id]

        if self.sat_id not in prev or self.sat_id not in curr:
            self.max_doppler = 0.0
            return

        d_prev = prev[self.sat_id].range
        d_curr = curr[self.sat_id].range
        radial_speed = (d_curr - d_prev) / self.traj_time_step
        self.max_doppler = abs(radial_speed) * self.carrier_freq / c

    def _update_fspl(self):
        d_km = self.distance / 1000.0
        f_mhz = self.carrier_freq / 1e6
        self.fspl = -32.5 - 20 * np.log10(d_km) - 20 * np.log10(f_mhz)

    def _update_time_correlation(self):
        self.rho = j0(2 * np.pi * abs(self.max_doppler) * self.time_step)

    # ===============================================================
    # 多径：固定 PDP + Doppler 分布
    # ===============================================================

    def _update_doppler_shifts(self):
        shifts = np.zeros(self.num_taps)
        shifts[0] = self.max_doppler
        if self.num_taps > 1:
            ang = np.linspace(0, 2 * np.pi, self.num_taps)[:-1]
            shifts[1:] = self.max_doppler * np.cos(ang[1:])
        self.doppler_shifts = shifts

    def _generate_static_parameters(self):
        # 频率网格
        self.freqs = np.arange(-self.bandwidth / 2 + self.scs / 2,
                               self.bandwidth / 2, self.scs)

        # tap 延迟
        self.tap_delays = np.zeros(self.num_taps)
        if self.num_taps > 1:
            self.tap_delays[1:] = np.sort(
                np.random.uniform(0, self.max_delay_spread, self.num_taps - 1)
            )

        decay = 3.0 / self.max_delay_spread
        pdp = np.exp(-decay * self.tap_delays)
        pdp[0] = self.total_power_tap0
        self.pdp_normalized = pdp / np.sum(pdp)

        self.v0 = np.sqrt(self.omega) * np.exp(1j * np.random.uniform(0, 2 * np.pi))
        self.sigma0 = np.sqrt(self.b_ssr)

        self._update_doppler_shifts()

    # ===============================================================
    # 小尺度：AR(1)
    # ===============================================================

    def initialize_channel(self):
        self.current_slot = 0
        self._update_satellite_parameters(0)
        self._update_doppler_shifts()

        # 初始化 tap scattering（NLOS 部分）
        self.tap_scattering = np.zeros(self.num_taps, dtype=complex)
        # tap0
        self.tap_scattering[0] = self.sigma0 * (np.random.randn() + 1j * np.random.randn())
        # 其他 tap
        for i in range(1, self.num_taps):
            scale = np.sqrt(self.pdp_normalized[i] / 2)
            self.tap_scattering[i] = scale * (np.random.randn() + 1j * np.random.randn())

    def step_channel(self):
        self.current_slot += 1
        if self.current_slot >= self.num_time_slots:
            raise StopIteration

        # 更新轨迹 & 大尺度
        self._update_satellite_parameters(self.current_slot)
        self._update_doppler_shifts()

        # 如果不可见：小尺度直接清零（信道 ≈ -∞）
        if not self.visible_now:
            self.tap_scattering[:] = 0.0j
            return

        # AR(1)
        noise_scaling = np.sqrt(max(0, 1 - self.rho ** 2))
        new_sc = np.zeros(self.num_taps, dtype=complex)

        # tap0
        noise0 = self.sigma0 * noise_scaling * (np.random.randn() + 1j * np.random.randn())
        new_sc[0] = self.rho * self.tap_scattering[0] + noise0

        # 其他 taps
        for i in range(1, self.num_taps):
            scale = np.sqrt(self.pdp_normalized[i] / 2)
            noise = scale * noise_scaling * (np.random.randn() + 1j * np.random.randn())
            new_sc[i] = self.rho * self.tap_scattering[i] + noise

        self.tap_scattering = new_sc

    # ===============================================================
    # 输出：RB 级 CNR = |h|^2 / σ^2
    # ===============================================================

    def get_rb_cnr(self):
        if self.current_slot < 0:
            raise RuntimeError("channel not initialized")

        if not self.visible_now:
            # 不可见：直接返回一个非常低的值（例如 -200 dB）
            return np.full(self.num_rb, -200.0)

        # assemble taps
        tap_gains = np.zeros(self.num_taps, dtype=complex)
        tap_gains[0] = self.v0 + self.tap_scattering[0]
        for i in range(1, self.num_taps):
            tap_gains[i] = self.tap_scattering[i]

        # 频率响应
        h = np.zeros(len(self.freqs), dtype=complex)
        for idx_f, f in enumerate(self.freqs):
            acc = 0.0j
            for k in range(self.num_taps):
                f_shift = f + self.doppler_shifts[k]
                phase = 2 * np.pi * f_shift * self.tap_delays[k]
                acc += tap_gains[k] * np.exp(-1j * phase)
            h[idx_f] = acc

        # 大尺度增益
        gain_db = self.fspl + self.tx_antenna_gain + self.rx_antenna_gain
        gain_lin = 10 ** (gain_db / 10)
        h_eff = h * np.sqrt(gain_lin)

        # 噪声功率
        noise_power_dbm = self.noise_psd_dbmhz + 10 * np.log10(self.scs) + self.noise_figure_db
        noise_power_w = 10 ** ((noise_power_dbm - 30) / 10)

        cnr_sub = (np.abs(h_eff) ** 2) / (noise_power_w + 1e-30)

        cnr_rb = np.zeros(self.num_rb)
        for rb in range(self.num_rb):
            s = rb * self.num_sc_per_rb
            e = s + self.num_sc_per_rb
            cnr_rb[rb] = np.mean(cnr_sub[s:e])

        return 10 * np.log10(cnr_rb + 1e-20)


# ===============================================================
# 示例：画 t=0 的频率选择性谱形
# ===============================================================

def get_initial_rb_cnr(trajectory_file, usr_id, sat_id):
    ch = TimeVaryingSatelliteChannel(
        total_time=200.0,
        time_step=0.01,
        trajectory_file=trajectory_file,
        usr_id=usr_id,
        sat_id=sat_id,
        traj_time_step=0.1
    )
    ch.initialize_channel()
    return ch.get_rb_cnr()


def plot_user_sat_three_slots(trajectory_file,
                              usr_id,
                              sat_id,
                              slot_indices,
                              time_step=0.01,
                              traj_time_step=0.1,
                              bandwidth=20e6,
                              scs=15e3):
    """
    绘制同一 (usr_id, sat_id) 在给定三个时隙下的 RB 级 CNR 曲线。

    参数：
    - trajectory_file : 轨迹 pkl 文件路径
    - usr_id, sat_id  : 用户和卫星 ID
    - slot_indices    : 长度为 3 的列表，例如 [0, 5, 10]，代表第几个信道时隙
    - time_step       : 信道时间粒度，单位秒（默认 0.01s = 10ms）
    - traj_time_step  : 轨迹时间粒度，单位秒（默认 0.1s）
    """

    assert len(slot_indices) == 3, "slot_indices 必须是 3 个时隙索引"
    slot_indices = sorted(slot_indices)
    max_slot = slot_indices[-1]

    # 自动设置 total_time，保证能覆盖到最大时隙
    total_time = (max_slot + 1) * time_step

    # 创建信道实例
    ch = TimeVaryingSatelliteChannel(
        total_time=total_time,
        time_step=time_step,
        trajectory_file=trajectory_file,
        usr_id=usr_id,
        sat_id=sat_id,
        traj_time_step=traj_time_step
    )
    # 可选：如果你想改带宽/SCS，也可以传进来
    ch.bandwidth = bandwidth
    ch.scs = scs
    ch.num_rb = int(bandwidth // (scs * ch.num_sc_per_rb))
    ch.total_subcarriers = ch.num_rb * ch.num_sc_per_rb
    ch._generate_static_parameters()  # 重新生成频率网格等

    ch.initialize_channel()

    # 存结果：slot_idx -> cnr
    cnr_dict = {}
    current_slot = 0

    # slot 0 先取一次
    if 0 in slot_indices:
        cnr_dict[0] = ch.get_rb_cnr()

    # 一路 step 到最大 slot，并在指定时刻记录 CNR
    while current_slot < max_slot:
        ch.step_channel()
        current_slot += 1
        if current_slot in slot_indices:
            cnr_dict[current_slot] = ch.get_rb_cnr()

    # 取出三条
    s0, s1, s2 = slot_indices
    cnr0 = cnr_dict[s0]
    cnr1 = cnr_dict[s1]
    cnr2 = cnr_dict[s2]

    # 画图
    x = np.arange(len(cnr0))
    plt.figure(figsize=(10, 5))
    plt.plot(x, cnr0, label=f"slot {s0} (t={s0 * time_step * 1000:.0f} ms)", linewidth=2.0)
    plt.plot(x, cnr1, label=f"slot {s1} (t={s1 * time_step * 1000:.0f} ms)", linewidth=1.7, linestyle="--")
    plt.plot(x, cnr2, label=f"slot {s2} (t={s2 * time_step * 1000:.0f} ms)", linewidth=1.7, linestyle=":")
    plt.xlabel("RB index")
    plt.ylabel(r"$|h|^2 / \sigma^2$ (dB)")
    plt.title(f"User {usr_id} – Sat {sat_id}: RB CNR at slots {s0}, {s1}, {s2}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"user{usr_id}_sat{sat_id}_slots{s0}_{s1}_{s2}_rb_cnr.png", dpi=300)

    # 顺便输出相关系数，方便你 sanity check
    def corr(a, b):
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)
        return np.dot(a0, b0) / (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12)

    print(f"corr(slot{s0}, slot{s1}) = {corr(cnr0, cnr1)}")
    print(f"corr(slot{s0}, slot{s2}) = {corr(cnr0, cnr2)}")
    print(f"corr(slot{s1}, slot{s2}) = {corr(cnr1, cnr2)}")

    return cnr0, cnr1, cnr2


if __name__ == "__main__":
    TRAJ = "sat-data/visibles_t_u-46users.pkl"

    # 1. 确认对同一颗卫星接入时 不同用户的信道的独立性
    # cnr1 = get_initial_rb_cnr(TRAJ, usr_id=0, sat_id=38)
    # cnr2 = get_initial_rb_cnr(TRAJ, usr_id=24, sat_id=38)

    # x = np.arange(len(cnr1))
    # plt.figure(figsize=(10, 5))
    # plt.plot(x, cnr1, label="User0-Sat38")
    # plt.plot(x, cnr2, label="User24-Sat1688")
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.legend()
    # plt.xlabel("RB index")
    # plt.ylabel(r"$|h|^2 / \sigma^2$ (dB)")
    # plt.tight_layout()
    # plt.show()


    # 2. 确认卫星-用户信道在不同slot的相关性
    cnr0, cnr5, cnr10 = plot_user_sat_three_slots(
        TRAJ,
        usr_id=0,
        sat_id=38,
        slot_indices=[0, 10, 30],   # 这里随便指定三个时隙
        time_step=0.01,
        traj_time_step=0.1
    )

