from tkinter import NO
import numpy as np
from commModule import leoSat
from commModule.apps import CommApplication
from commModule.slice import Slice
from commModule.leoSat import LEOSatellite
from commModule.UserTerminal import UserTerminal
from commModule.UserTerminal import Position
from commModule.UserTerminal import Packet
from concurrent.futures import ThreadPoolExecutor

def dbm_to_watts(dbm):
    """
    Convert dBm to watts.

    Parameters:
        dbm (float): Power in dBm.

    Returns:
        float: Power in watts.
    """
    return 10 ** ((dbm - 30) / 10)

class MacScheduler:

    def __init__(self, mac_slot_duration=10.0, slice_slot_duration=100.0, policy="ADMM"):
        self.mac_slot_duration = mac_slot_duration
        self.slice_slot_duration = slice_slot_duration
        self.policy = policy
        self.current_time = 0  # 当前时间ms

        self.admm_vars = {}
        self.debug = True


    def round_robin_schedule(self, satellite: LEOSatellite, action_dict: dict):
        """
        Schedule RBs and power for all slices and apps on a specific satellite using Round Robin (RR) strategy,
        considering slice bandwidth and power limits.

        Parameters:
            satellite (LEOSatellite):
                The satellite object for which resources are being allocated.

            current_time (int):
                The current time slot number.

            action_dict (dict):
                A dictionary containing bandwidth and power limits for the satellite's slices.
                Keys include:
                    - "embb_bd" (int): Number of RBs allocated to the eMBB slice.
                    - "urllc_bd" (int): Number of RBs allocated to the URLLC slice.
                    - "embb_power" (float): Power ratio allocated to the eMBB slice (0~1).
                    - "urllc_power" (float): Power ratio allocated to the uRLLC slice (0~1).
                    - "ue_targets" (list): List of user terminal IDs connected to the satellite.

        Returns:
            rb_allocation (dict):
                A dictionary mapping app IDs to their allocated RB indices.

            power_allocation (dict):
                A dictionary mapping app IDs to their allocated power values.
        """
        # 初始化分配结果
        rb_allocation = {}
        power_allocation = {}

        # 获取卫星的总资源块（RB）和最大功率
        total_rbs = list(range(satellite._rbs))  # 假设 _rbs 是一个整数，表示总的 RB 数量
        max_power = dbm_to_watts(satellite.max_tx_power)  # 卫星的最大功率

        # 从 action_dict 中获取切片的带宽和功率限额
        embb_rb_limit = action_dict["embb_bd"]
        urllc_rb_limit = action_dict["urllc_bd"]
        embb_power_ratio = action_dict["embb_power"]
        urllc_power_ratio = action_dict["urllc_power"]

        # 归一化功率比例
        total_power_ratio = embb_power_ratio + urllc_power_ratio
        if( total_power_ratio == 0):
            embb_power_ratio = 0
            urllc_power_ratio = 0
        else:
            embb_power_ratio /= total_power_ratio
            urllc_power_ratio /= total_power_ratio

        # 获取切片
        slices = satellite._slices
        embb_slice = next((s for s in slices if s.category == "eMBB"), None)
        urllc_slice = next((s for s in slices if s.category == "uRLLC"), None)

        # 初始化每个切片的分配
        slice_rb_allocation = {"eMBB": 0, "uRLLC": 0}
        slice_power_allocation = {"eMBB": 0, "uRLLC": 0}

        # 初始化每个应用的分配
        for slice_obj in slices:
            for app in slice_obj.apps.values():
                rb_allocation[app.ID] = []
                power_allocation[app.ID] = []

        # Round Robin 分配 RB
        embb_apps = list(embb_slice.apps.values()) if embb_slice else []
        urllc_apps = list(urllc_slice.apps.values()) if urllc_slice else []
        for rb in total_rbs:
            allocated = False
            # 尝试给 eMBB 分配（前提：该 slice 存在、未超过限额且有 app）
            if embb_slice and slice_rb_allocation["eMBB"] < embb_rb_limit and embb_apps:
                app = embb_apps[slice_rb_allocation["eMBB"] % len(embb_apps)]
                rb_allocation[app.ID].append(rb)
                slice_rb_allocation["eMBB"] += 1
                allocated = True

            # 如果 eMBB 未被分配，则尝试给 uRLLC（前提同上）
            if not allocated and urllc_slice and slice_rb_allocation["uRLLC"] < urllc_rb_limit and urllc_apps:
                app = urllc_apps[slice_rb_allocation["uRLLC"] % len(urllc_apps)]
                rb_allocation[app.ID].append(rb)
                slice_rb_allocation["uRLLC"] += 1
                allocated = True

        # 平均分配功率
        for slice_type, slice_obj in [("eMBB", embb_slice), ("uRLLC", urllc_slice)]:
            if slice_obj:
                power_ratio = embb_power_ratio if slice_type == "eMBB" else urllc_power_ratio
                slice_power_limit = power_ratio * max_power  # 切片的功率限额
                for app in slice_obj.apps.values():
                    num_rbs = len(rb_allocation[app.ID])
                    if num_rbs > 0:
                        power_per_rb = slice_power_limit / (embb_rb_limit if slice_type == "eMBB" else urllc_rb_limit)
                        power_allocation[app.ID] = [power_per_rb] * num_rbs

        # print(f"Satellite {satellite.ID} RB Allocation: {rb_allocation}") 
        # print(f"Satellite {satellite.ID} Power Allocation: {power_allocation}")
        return rb_allocation, power_allocation

    def admm_schedule(self, satellite: LEOSatellite, action_dict: dict, snr_dict: dict):
        """
        Schedule RBs and power for all slices and apps on a specific satellite using ADMM-based optimization.

        Parameters:
            satellite (LEOSatellite):
                The satellite object for which resources are being allocated.

            action_dict (dict):
                A dictionary containing bandwidth and power limits for the satellite's slices.

            snr_dict (dict):
                A dictionary where the key is the user terminal ID, and the value is the SNR list for each RB.

        Returns:
            rb_allocation (dict):
                A dictionary mapping app IDs to their allocated RB indices.

            power_allocation (dict):
                A dictionary mapping app IDs to their allocated power values.
        """
        import cvxpy as cp
        import numpy as np


        # --- 工具函数 ---
        def compute_throughput(x_vec, snr_vec, p_vec, rb_bandwidth):
            throughput_bps = np.sum(x_vec * rb_bandwidth * np.log2(1 + np.clip(snr_vec * p_vec, 1e-12, None)))
            return throughput_bps / 1e6  # 转换为 Mbps

        def compute_delay(data_size, throughput_mbps):
            throughput_bps = throughput_mbps * 1e6  # 转换为 bps
            return (data_size / (throughput_bps + 1e-6)) * 1e3  # 转换为 ms

        def user_satisfaction(throughput, delay, Tmin, Dmax, beta_t=5, beta_d=5):
            sat_t = 1 / (1 + np.exp(-np.clip(beta_t * (throughput - Tmin), -700, 700)))
            sat_d = 1 / (1 + np.exp(np.clip(beta_d * (delay - Dmax), -700, 700)))
            return sat_t + sat_d

        def compute_weights(queue_lengths, deadlines, snr_vec, alpha=0.5):
            K = len(queue_lengths)  # 应用数量
            RB_COUNT = len(snr_vec)  # RB 总数
            rb_per_app = RB_COUNT // K  # 每个应用分配的 RB 数量（假设均匀分配）

            # 计算队列紧迫度（队列长度 / 剩余时间）
            queue_urgency = queue_lengths / deadlines

            # 归一化队列紧迫度（越大越紧迫）
            queue_urgency_norm = queue_urgency / np.max(queue_urgency)

            # 为每个应用计算平均信噪比
            app_snr = np.array([np.mean(snr_vec[i * rb_per_app:(i + 1) * rb_per_app]) for i in range(K)])

            # 归一化信噪比（越小越差）
            snr_norm = 1 - (app_snr / np.max(app_snr))  # 信噪比越小，权重越大

            # 计算最终权重（每个应用一个权重）
            weights = alpha * queue_urgency_norm + (1 - alpha) * snr_norm
            return weights

        # --- X 子问题（布尔 + 可行投影） ---
        def make_x_solver(app_snr_vec, p_vec, rho, v_param, data_size, Tmin, Dmax, cost_r, cost_p, rb_bandwidth, weights, beta_t=5, beta_d=5):
            RB_COUNT = len(app_snr_vec)
            x_var = cp.Variable(RB_COUNT, boolean=True)

            throughput_val = compute_throughput(np.ones(RB_COUNT), app_snr_vec, p_vec, rb_bandwidth)
            T_grad = (beta_t * np.exp(-np.clip(beta_t * (throughput_val - Tmin), -700, 700))) / \
                    ((1 + np.exp(-np.clip(beta_t * (throughput_val - Tmin), -700, 700)))**2)
            throughput_grad = np.log1p(app_snr_vec * p_vec) * rb_bandwidth / 1e6  # 转换为 Mbps
            lin_satisfaction = weights * cp.sum(cp.multiply(x_var, T_grad * throughput_grad))  # 直接乘以标量权重

            # 目标
            obj = cp.Maximize(lin_satisfaction - cost_r * cp.sum(x_var) - cost_p * cp.sum(cp.multiply(x_var, p_vec))
                            - 0.5 * rho * cp.sum_squares(x_var - v_param))

            # # RB 可行性约束（松弛到 [0,1] 投影，后面裁剪）
            # constraints = [x_var >= 0, x_var <= 1]

            def debug_print():
                print(f"[DEBUG] X Solver:")
                print(f"  Satisfaction: {lin_satisfaction.value}")
                print(f"  RB Cost: {cost_r * np.sum(x_var.value)}")
                print(f"  Power Cost: {cost_p * np.sum(np.multiply(x_var.value, p_vec))}")
                print(f"  Quadratic Penalty: {0.5 * rho * np.sum((x_var.value - v_param)**2)}")

            prob = cp.Problem(obj)

            def solver():
                prob.solve(solver="ECOS_BB", verbose=False)
                val = x_var.value
                if val is None:
                    val = np.zeros(RB_COUNT)
                if self.debug:
                    debug_print()
                return np.clip(np.round(val), 0, 1)
            return solver

        # --- P 子问题（连续 + SCA线性化） ---
        def make_p_solver(app_snr_vec, x_vec, rho, v_param, data_size, Tmin, Dmax, cost_r, cost_p, rb_bandwidth, p0, weights, P_max_per_RB=1, beta_t=5, beta_d=5):
            RB_COUNT = len(app_snr_vec)
            p_var = cp.Variable(RB_COUNT, nonneg=True)

            # 使用 p0 线性化
            throughput_val = compute_throughput(x_vec, app_snr_vec, p0, rb_bandwidth)
            T_grad = (beta_t * np.exp(-np.clip(beta_t * (throughput_val - Tmin), -700, 700))) / \
                    ((1 + np.exp(-np.clip(beta_t * (throughput_val - Tmin), -700, 700)))**2)
            throughput_grad = x_vec * app_snr_vec / (1 + app_snr_vec * p0) * rb_bandwidth / 1e6  # 转换为 Mbps
            lin_satisfaction = weights * cp.sum(cp.multiply(throughput_grad * T_grad, p_var))  # 直接乘以标量权重

            # 约束：物理上限 & RB 未分配则功率为0
            constraints = [p_var >= 0, p_var <= P_max_per_RB * x_vec]

            obj = cp.Maximize(lin_satisfaction - cost_r * np.sum(x_vec) - cost_p * cp.sum(cp.multiply(x_vec, p_var))
                            - 0.5 * rho * cp.sum_squares(p_var - v_param))
            prob = cp.Problem(obj, constraints)


            def debug_print():
                print(f"[DEBUG] P Solver:")
                print(f"  Satisfaction: {lin_satisfaction.value}")
                print(f"  RB Cost: {cost_r * np.sum(x_vec)}")
                print(f"  Power Cost: {cost_p * np.sum(np.multiply(x_vec, p_var.value))}")
                print(f"  Quadratic Penalty: {0.5 * rho * np.sum((p_var.value - v_param)**2)}")
            prob = cp.Problem(obj)

            def solver():
                prob.solve(solver="ECOS", verbose=False)
                val = p_var.value
                if val is None:
                    val = np.zeros(RB_COUNT)
                if self.debug:
                    debug_print()
                return np.clip(val, 0, P_max_per_RB)
            return solver

        # --- Z 更新（投影可行域） ---
        def z_solver(X, P, Ux, Up, apps, slices, slice_rb_limits, slice_power_limits, rho, P_max_per_RB=1):
            K, RB_COUNT = X.shape
            Zx = cp.Variable((K, RB_COUNT), nonneg=True)  # 定义为非负连续变量布尔变量
            Zp = cp.Variable((K, RB_COUNT), nonneg=True)  # 定义为非负连续变量

            # 目标函数
            obj = cp.Minimize((rho / 2) * cp.sum_squares(X - Zx) + (rho / 2) * cp.sum_squares(P - Zp))

            constraints = []

            # RB 互斥约束 & 上界
            for j in range(RB_COUNT):
                constraints.append(cp.sum(Zx[:, j]) <= 1)  # 每个 RB 最多分配给一个应用
            constraints.append(Zx <= 1)
            constraints.append(Zx >= 0)

            # 切片限制
            for slice_idx, slice_obj in enumerate(slices):
                idxs = [i for i in range(K) if apps[i].ID in slice_obj.apps]  # 确定属于该切片的应用索引
                if idxs:  # 如果该切片有应用
                    constraints.append(cp.sum(Zx[idxs, :]) <= slice_rb_limits[slice_obj._category])  # RB 限制
                    constraints.append(cp.sum(Zp[idxs, :]) <= slice_power_limits[slice_obj._category])  # 功率限制

            # 功率限制
            for j in range(RB_COUNT):
                constraints.append(cp.sum(Zp[:, j]) <= P_max_per_RB)  # 每个 RB 的功率限制

            # 保证功率分配与 RB 分配一致
            constraints.append(Zp >= 0)
            constraints.append(Zp <= P_max_per_RB * Zx)

            # 求解问题
            prob = cp.Problem(obj, constraints)
            prob.solve(solver="ECOS", verbose=False)

            Zx_val = Zx.value
            Zp_val = Zp.value
            if Zx_val is None:
                Zx_val = X.copy()
            if Zp_val is None:
                Zp_val = P.copy()
            Zx_val = np.clip(Zx_val, 0, 1)
            Zp_val = np.clip(Zp_val, 0, P_max_per_RB)
            return Zx_val, Zp_val

        # --- ADMM 主函数 ---
        def admm_sca_satellite_allocation(sat_id, apps, slices, snr_dict, data_sizes, Tmin, Dmax,
                                        cost_r=0.05, cost_p=0.05, slice_rb_limits=None, slice_power_limits=None,
                                        SCS=15, admm_inner_iters=20, rho=10.0,
                                        eps_abs=1e-3, eps_rel=1e-3, P_max_per_RB=1, beta_t=10, beta_d=10, alpha=0.5):
            K = len(apps)
            RB_COUNT = len(next(iter(snr_dict.values())))  # 假设所有 UE 的 SNR 列表长度相同
            RB_bandwidth = SCS * 12.0 * 1e3  # 每个 RB 的带宽，单位 Hz

            X = np.random.choice([0, 1], size=(K, RB_COUNT)).astype(bool).astype(float)
            P = np.random.uniform(0, P_max_per_RB, (K, RB_COUNT))
            Zx = X.copy()
            Zp = P.copy()
            Ux = np.zeros_like(X)
            Up = np.zeros_like(P)

            if sat_id in self.admm_vars:
                X = self.admm_vars[sat_id]['X']
                P = self.admm_vars[sat_id]['P']
                Zx = self.admm_vars[sat_id]['Zx']
                Zp = self.admm_vars[sat_id]['Zp']
                Ux = self.admm_vars[sat_id]['Ux']
                Up = self.admm_vars[sat_id]['Up']

            # 计算权重
            deadlines = np.array([Dmax[i] for i in range(K)])  # 假设每个应用的 ddl 是 Dmax
            queue_lengths = data_sizes  # 假设队列长度等于数据量
            weights = compute_weights(queue_lengths, deadlines, np.concatenate(list(snr_dict.values())), alpha)

            for it in range(admm_inner_iters):
                Zx_old = Zx.copy()
                Zp_old = Zp.copy()

                # X 更新
                for i, app in enumerate(apps):
                    app_snr_vec = snr_dict[app.ID]  # 获取当前应用的 SNR 列表
                    x_solver = make_x_solver(app_snr_vec, P[i, :], rho, Zx[i, :] - Ux[i, :],
                                            data_sizes[i], Tmin[i], Dmax[i], cost_r, cost_p, RB_bandwidth, weights[i], beta_t, beta_d)
                    X[i, :] = x_solver()

                # P 更新
                for i, app in enumerate(apps):
                    app_snr_vec = snr_dict[app.ID]  # 获取当前应用的 SNR 列表
                    p_solver = make_p_solver(app_snr_vec, X[i, :], rho, Zp[i, :] - Up[i, :],
                                            data_sizes[i], Tmin[i], Dmax[i], cost_r, cost_p, RB_bandwidth,
                                            P[i, :], weights[i], P_max_per_RB, beta_t, beta_d)
                    P[i, :] = p_solver()
                # Z 更新
                Zx, Zp = z_solver(X, P, Ux, Up, apps, slices, slice_rb_limits, slice_power_limits, rho, P_max_per_RB)
                # U 更新
                Ux += X - Zx
                Up += P - Zp

                # 打印满意度
                total_sat = 0
                for i in range(K):
                    th = compute_throughput(Zx[i, :], snr_dict[apps[i].ID], Zp[i, :], RB_bandwidth)
                    dly = compute_delay(data_sizes[i], th)
                    total_sat += user_satisfaction(th, dly, Tmin[i], Dmax[i], beta_t, beta_d)
                avg_sat = total_sat / K

                # 检查收敛条件
                r_norm = np.sqrt(np.sum((X - Zx) ** 2) + np.sum((P - Zp) ** 2))
                s_norm = np.sqrt(np.sum((Zx - Zx_old) ** 2) + np.sum((Zp - Zp_old) ** 2))
                eps_pri = np.sqrt(K * RB_COUNT * 2) * eps_abs + eps_rel * max(np.linalg.norm(X, 'fro'), np.linalg.norm(Zx, 'fro'),
                                                                            np.linalg.norm(P, 'fro'), np.linalg.norm(Zp, 'fro'))
                eps_dual = np.sqrt(K * RB_COUNT * 2) * eps_abs + eps_rel * max(np.linalg.norm(Ux, 'fro'), np.linalg.norm(Up, 'fro'))
                if self.debug:
                    print(f"Sat {sat_id} Iter {it}: avg_satisfaction={avg_sat:.4f}, r_norm={r_norm:.4f}, s_norm={s_norm:.4f}, eps_pri={eps_pri:.4f}, eps_dual={eps_dual:.4f}")
                if r_norm < eps_pri and s_norm < eps_dual:
                    break
            
            self.admm_vars[sat_id] = {
                'X': X,
                'P': P,
                'Zx': Zx,
                'Zp': Zp,
                'Ux': Ux,
                'Up': Up,
            }
            return Zx, Zp

        # --- 调度逻辑 ---
        slices = satellite._slices
        apps = []
        for slice_obj in slices:
            apps.extend(slice_obj.apps.values())

        K = len(apps)

        if( K == 0):
            return {}, {}

        RB_COUNT = len(next(iter(snr_dict.values())))  # 假设所有 UE 的 SNR 列表长度相同
        slice_rb_limits = {
            "eMBB": action_dict["embb_bd"],
            "uRLLC": action_dict["urllc_bd"]
        }
        if action_dict["embb_power"] + action_dict["urllc_power"] > 0:
            slice_power_limits = {
                "eMBB": action_dict["embb_power"] / (action_dict["embb_power"] + action_dict["urllc_power"]) * dbm_to_watts(satellite.max_tx_power),
                "uRLLC": action_dict["urllc_power"] / (action_dict["embb_power"] + action_dict["urllc_power"]) * dbm_to_watts(satellite.max_tx_power)
            }
        else:
            slice_power_limits = {
                "eMBB": 0,
                "uRLLC": 0
            }


        data_sizes = np.array([app._packet_size for app in apps])
        Tmin = np.array([app._BW_u for app in apps])
        Dmax = np.array([app._Delay_u for app in apps])

        Zx, Zp = admm_sca_satellite_allocation(
            sat_id=satellite.ID,
            apps=apps,
            slices=slices,
            snr_dict=snr_dict,
            data_sizes=data_sizes,
            Tmin=Tmin,
            Dmax=Dmax,
            slice_rb_limits=slice_rb_limits,
            slice_power_limits=slice_power_limits,
            SCS=satellite.scs
        )

        # 转换为 rb_allocation 和 power_allocation 格式
        rb_allocation = {app.ID: np.where(Zx[i, :] > 0.5)[0].tolist() for i, app in enumerate(apps)}
        power_allocation = {app.ID: Zp[i, rb_allocation[app.ID]].tolist() for i, app in enumerate(apps)}
        # print("rb_allocation: ", rb_allocation)
        # print("power_allocation: ", power_allocation)

        return rb_allocation, power_allocation

    def schedule(self, leo_sats: list[LEOSatellite], action_dict: dict, UEs: dict, sat_snr_dict: dict):
        """
        leo_sats (list[LEOSatellite]):
            A list of all Low Earth Orbit satellites (LEOSatellite objects).
            Each satellite object includes its resources (e.g., bandwidth, RB count) and scheduling state.

        action_dict (dict):
            A dictionary describing the scheduling actions for the current time slot.
            Keys are satellite IDs, and values are dictionaries containing the following keys:
                - "embb_bd" (int): Number of RBs allocated to the eMBB slice.
                - "urllc_bd" (int): Number of RBs allocated to the URLLC slice.
                - "embb_power" (float): Power ratio allocated to the eMBB slice (0~1).
                - "urllc_power" (float): Power ratio allocated to the URLLC slice (0~1).
                - "ue_targets" (list): List of user terminal IDs connected to the satellite.

        UEs (dict):
            A dictionary containing all user terminals (UserTerminal objects).
            Keys are user terminal IDs, and values are the corresponding UserTerminal objects.
            Each user terminal object includes its connected satellite, application requirements (e.g., bandwidth, latency), etc.

        sat_snr_dict (dict):
            A dictionary predict the Signal-to-Noise Ratio (SNR) information for each satellite.
            Keys are satellite IDs, and values are dictionaries containing the following keys:
                - User terminal ID (int): The SNR information for the corresponding user.
                - Each user's SNR information is a dictionary where keys are time granularities (0~9), and values are lists of SNRs for each RB.
                Example:
                {
                    sat_id: {
                        ue_id: {
                            0: [snr_rb1, snr_rb2, ...],
                            1: [...],
                            ...
                        }
                    }
                }
        """
        def process_satellite(sat: LEOSatellite):
            """
            Process scheduling for a single satellite over 10 slots.

            Parameters:
                sat (LEOSatellite): The satellite to process.

            Returns:
                dict: The scheduling results for the satellite.
                    {
                        "success": True/False,
                        "allocation": {
                            slot_id: {
                                "rb_allocation": {app_id: [...]},
                                "power_allocation": {app_id: [...]}
                            }
                        }
                    }
            """
            allocation = {}
            try:
                for slot_id in range(10):  # 遍历 10 个时隙
                    # 获取该时隙的 SNR 数据
                    snr_dict = {}
                    for ue_id in action_dict[sat.ID]["ue_targets"]:
                        snr_dict[ue_id] = [10 ** (snr / 10) for snr in sat_snr_dict[sat.ID][ue_id][slot_id]]

                    # 根据 policy 调用调度方法
                    if self.policy == "RR":
                        rb_allocation, power_allocation = self.round_robin_schedule(sat, action_dict[sat.ID])
                    elif self.policy == "ADMM":
                        rb_allocation, power_allocation = self.admm_schedule(sat, action_dict[sat.ID], snr_dict)
                    else:
                        raise ValueError(f"Unknown scheduling policy: {self.policy}")

                    # 保存结果
                    allocation[slot_id] = {
                        "rb_allocation": rb_allocation.copy(),
                        "power_allocation": power_allocation.copy(),
                    }

                # 清空 ADMM 变量
                self.admm_vars[sat.ID] = {}

                return {"success": True, "allocation": allocation}

            except Exception as e:
                print(f"Error in scheduling for satellite {sat.ID}: {e}")
                return {"success": False, "allocation": {}}

        # 创建线程池，线程数量等于卫星数量
        submitted_sats = [sat for sat in leo_sats if "ue_targets" in action_dict.get(sat.ID, {})]
        if not submitted_sats:
            return {"success": True, "allocation": {}}

        max_workers = max(1, len(submitted_sats))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_satellite, sat) for sat in submitted_sats]

            # 收集结果（使用 submitted_sats 与 futures 一一配对，避免错位）
            success = True
            allocation = {}
            for future, sat in zip(futures, submitted_sats):
                result = future.result()
                if not result["success"]:
                    success = False
                    allocation = {}  # 如果任何一个卫星失败，清空分配结果
                    break
                allocation[sat.ID] = result["allocation"]
 
        return {"success": success, "allocation": allocation}

    def statistics_summary(self, leo_sats: list[LEOSatellite], action_dict: dict, UEs: dict, sat_snr_dict: dict, allocation: dict):
        """
        Collect statistics about the scheduler's results and performance metrics.

        This function calls `round_robin_schedule` 100 times to perform RB allocation for all satellites and their connected apps.
        It returns a dictionary where the key is the app ID, and the value is another dictionary containing:
            - The satellite ID to which the app is connected.
            - The slice type the app belongs to (e.g., "eMBB" or "uRLLC").
            - Whether the app meets both bandwidth and latency requirements (both must be satisfied to return `True`).
            - The user satisfaction value, calculated using a sigmoid function.

        Parameters:
            leo_sats (list[LEOSatellite]):
                A list of all Low Earth Orbit satellites (LEOSatellite objects).
                Each satellite object includes its resources (e.g., bandwidth, RB count) and scheduling state.

            action_dict (dict):
                A dictionary describing the scheduling actions for the current time slot.
                Keys are satellite IDs, and values are dictionaries containing the following keys:
                    - "embb_bd" (int): Number of RBs allocated to the eMBB slice.
                    - "urllc_bd" (int): Number of RBs allocated to the URLLC slice.
                    - "embb_power" (float): Power ratio allocated to the eMBB slice (0~1).
                    - "urllc_power" (float): Power ratio allocated to the URLLC slice (0~1).
                    - "ue_targets" (list): List of user terminal IDs connected to the satellite.

            UEs (dict):
                A dictionary containing all user terminals (UserTerminal objects).
                Keys are user terminal IDs, and values are the corresponding UserTerminal objects.
                Each user terminal object includes its connected satellite, application requirements (e.g., bandwidth, latency), etc.

            sat_snr_dict (dict):
                A dictionary describing the Signal-to-Noise Ratio (SNR) information for each satellite.
                Keys are satellite IDs, and values are dictionaries containing the following keys:
                    - User terminal ID (int): The SNR information for the corresponding user.
                    - Each user's SNR information is a dictionary where keys are time granularities (0~9), and values are lists of SNRs for each RB.
                    Example:
                    {
                        sat_id: {
                            ue_id: {
                                0: [snr_rb1, snr_rb2, ...],
                                1: [...],
                                ...
                            }
                        }
                    }

        Returns:
            dict:
                A dictionary where the key is the app ID, and the value is another dictionary with the following keys:
                    - "satellite_id" (int): The ID of the satellite to which the app is connected.
                    - "slice_type" (str): The type of slice the app belongs to (e.g., "eMBB" or "uRLLC").
                    - "qos_met" (bool): Whether the app meets both bandwidth and latency requirements.
                    - "t_satisfaction" (float): The satisfaction level of the app's transmission requirements.
                    - "d_satisfaction" (float): The satisfaction level of the app's delay requirements.
        """
        app_results = {}  # 存储每个 app 的统计结果
        slice_results = {}  # 存储每个 slice 的统计结果

        # 初始化每个 app 的统计数据
        app_bandwidth_allocation = {ue.app.ID: 0 for ue in UEs.values()}  # 累计带宽分配
        app_throughput_allocation = {ue.app.ID: 0 for ue in UEs.values()}  # 累计吞吐量分配
        app_transmitted_bits = {ue.app.ID: 0 for ue in UEs.values()}  # 累计传输的比特数

        # 初始化每个切片的统计数据
        slice_transmitted_bits = {}  # 存储每个切片的总传输比特数
        for sat in leo_sats:
            for slice_obj in sat.slices:
                slice_key = f"{sat.ID}-{slice_obj.category}"
                slice_transmitted_bits[slice_key] = 0

        # 调度 10 次（slice_slot_duration / mac_slot_duration）
        num_schedules = int(self.slice_slot_duration / self.mac_slot_duration)
        for slot_id in range(num_schedules):
            # 每个 slot 全局起始时间（各卫星在该 slot 使用该时间的副本）
            slot_start_time = self.current_time
            for sat in leo_sats:
                if "ue_targets" not in action_dict[sat.ID]:
                    continue
                    
                # 调用 schedule 获取 RB 和功率分配
                rb_allocation = allocation[sat.ID][slot_id]["rb_allocation"]
                power_allocation = allocation[sat.ID][slot_id]["power_allocation"]

                # # === 新增：按 ADMM 内部一致的方式计算每个用户的满意度，便于对比 debug ===
                # # 与 admm 中一致的工具函数实现（简化，局部使用）
                # RB_bandwidth = sat.scs * 12.0 * 1e3  # 每个 RB 的带宽（Hz）
                # beta_t, beta_d = 5, 5
                # for app_id, rb_indices in rb_allocation.items():
                #     # 获取对应的功率列表（可能为空）
                #     p_list = power_allocation.get(app_id, [])
                #     # 计算吞吐量（逐 RB 累加），保持与 admm 中 compute_throughput 相同的公式
                #     throughput_bps = 0.0
                #     snr_list = sat_snr_dict[sat.ID][app_id][slot_id]
                #     for rb_idx, p in zip(rb_indices, p_list):
                #         snr_linear = 10 ** (snr_list[rb_idx] / 10)
                #         rb_rate = RB_bandwidth * np.log2(1 + snr_linear * p)  # bit/s
                #         throughput_bps += rb_rate
                #     throughput_mbps = throughput_bps / 1e6

                #     # 计算延迟（与 admm 中 compute_delay 保持一致）
                #     app_obj = UEs[app_id].app
                #     data_size = app_obj._packet_size  # 与 admm 使用的一致
                #     delay_ms = (data_size / (throughput_bps + 1e-6)) * 1e3

                #     # 计算满意度（与 admm 内部 user_satisfaction 保持一致）
                #     Tmin = app_obj._BW_u
                #     Dmax = app_obj._Delay_u
                #     sat_t = 1 / (1 + np.exp(-np.clip(beta_t * (throughput_mbps - Tmin), -700, 700)))
                #     sat_d = 1 / (1 + np.exp(np.clip(beta_d * (delay_ms - Dmax), -700, 700)))
                #     sat_val = sat_t + sat_d

                #     # 简单打印用于对比（可去掉或改成收集到结构中）
                #     print(f"[DEBUG] sat{sat.ID} slot{slot_id} app{app_id} -> th={throughput_mbps:.6f}Mbps delay={delay_ms:.3f}ms sat_admm_calc={sat_val:.6f}")

                # 遍历每个 app 的分配结果
                for app_id, rb_indices in rb_allocation.items():

                    app_time_ms = slot_start_time  # 每个应用使用卫星的时间副本进行计算
                    
                    ue = UEs[app_id]  # 获取对应的用户终端
                    app = ue.app
                    slice_type = None

                    # 获取 app 所属的切片类型
                    for slice_obj in sat.slices:
                        if app_id in slice_obj.apps:
                            slice_type = slice_obj.category
                            break

                    # 计算分配的带宽份额
                    rb_bandwidth = 12 * sat.scs * 1e3  # 单位 Hz
                    app_bandwidth_allocation[app_id] += rb_bandwidth * len(rb_indices)

                    # 计算当前时间段的 SNR 列表索引
                    snr_list = sat_snr_dict[sat.ID][ue.ID][slot_id]

                    throughput = 0
                    for rb_idx, power in zip(rb_indices, power_allocation[app_id]):
                        snr_linear = 10 ** (snr_list[rb_idx] / 10)
                        rb_rate = rb_bandwidth * np.log2(1 + snr_linear * power)  # 单位 bit/s
                        throughput += rb_rate
                    app_throughput_allocation[app_id] += throughput
                    
                    
                    for _ in range(int(self.mac_slot_duration)):
                        # 从 buffer 中传输数据包
                        buffer_empty = False
                        remaining_throughput = throughput / 1000  # 单位 bit
                        while remaining_throughput > 0 and not buffer_empty and not ue.buffer.empty():
                            packet = ue.buffer.queue[0]  # 获取队头数据包
                            if(app_time_ms < packet.gen_time):
                                buffer_empty = True
                                continue
                            if packet.start_transmission_time is None:
                                packet.start_transmission_time = app_time_ms  # 设置开始传输时间

                            remaining_packet_bits = (packet.size - packet.transmitted_size) * 8
                            if remaining_throughput >= remaining_packet_bits:
                                # 数据包完全传输
                                remaining_throughput -= remaining_packet_bits
                                app_transmitted_bits[app_id] += remaining_packet_bits
                                packet.transmitted_size = packet.size
                                ue.buffer.get()  # 从 buffer 中移除数据包
                                # print(f"Time {self.current_time} ms: UE {ue.ID} transmitted packet of size {packet.size} bytes.")

                                # 计算时延
                                queue_delay = packet.start_transmission_time - packet.gen_time
                                transmission_delay = app_time_ms - packet.start_transmission_time
                                total_delay = queue_delay + transmission_delay
                                ue.delay_pkt.append(total_delay)
                            else:
                                # 数据包部分传输
                                packet.transmitted_size += remaining_throughput / 8
                                app_transmitted_bits[app_id] += remaining_throughput
                                remaining_throughput = 0

                        app_time_ms += 1 # 更新应用时间


            self.current_time += self.mac_slot_duration # 更新全局 slot 起始时间
            

        for sat in leo_sats:
            for slice in sat._slices:
                slice_type = slice._category
                slice_key = f"{sat.ID}-{slice_type}"
                for app_id in slice.apps.keys():
                    slice_key = f"{sat.ID}-{slice_type}"
                    slice_transmitted_bits[slice_key] += app_transmitted_bits[app_id]

        for ue in UEs.values():
            app = ue.app
            # print(app)
            slice_type = None
            sat = ue.connect_satellite
            if sat is None:
                app_results[app.ID] = {
                    "satellite_id": None,
                    "slice_type": None,
                    "avg_rb_num": 0.0,
                    "avg_transmit_rate": 0.0,
                    "qos_met": False,
                    "throughput_satisfaction": 0.0,
                    "delay_satisfaction": 0.0
                }

                continue
            for slice_obj in sat._slices:
                if app.ID in slice_obj.apps:
                    slice_type = slice_obj.category
                    break

            # 获取app的平均到达率
            avg_transmit_rate = app_transmitted_bits[app.ID] / self.slice_slot_duration * 1e3  # 转换为 bps

            # 计算带宽数量
            avg_bandwidth = app_bandwidth_allocation[app.ID] / num_schedules # 平均带宽
            avg_rb_num = avg_bandwidth / (12 * sat.scs * 1e3)  # 平均 RB 数
            # 计算吞吐与时延
            avg_throughput = app_throughput_allocation[app.ID] / num_schedules  # 平均吞吐量
            avg_throughput_mbps = avg_throughput / 1e6  # 转换为 Mbps
            avg_delay = np.mean(ue.delay_pkt) if ue.delay_pkt else float('inf')  # 平均时延
            qos_met = avg_throughput_mbps >= app._BW_u and avg_delay <= app._Delay_u
            # print(f"App {app.ID} (Slice: {slice_type}) - Avg RBs: {avg_rb_num:.2f}, Avg Throughput: {avg_throughput_mbps:.6f} Mbps, Avg Delay: {avg_delay:.3f} ms, QoS Met: {qos_met}")

            # 计算用户满意度
            Tmin = app._BW_u
            Dmax = app._Delay_u
            t_satisfaction , d_satisfaction = self.user_satisfaction(avg_throughput_mbps, avg_delay, Tmin, Dmax)

            # 保存结果
            app_results[app.ID] = {
                "satellite_id": ue.get_connect_sat_id(),
                "slice_type": slice_type,
                "avg_rb_num": avg_rb_num,
                "avg_transmit_rate": avg_transmit_rate,
                "qos_met": qos_met,
                "throughput_satisfaction": t_satisfaction,
                "delay_satisfaction": d_satisfaction
            }
                        
        # 计算每个切片的业务到达率
        for slice_key, transmitted_bits in slice_transmitted_bits.items():
            slice_results[slice_key] = transmitted_bits / self.slice_slot_duration * 1e3  # 转换为 bps

        self.admm_vars = {}  # 清除 ADMM 变量
        return app_results, slice_results

    def user_satisfaction(self, throughput, delay, Tmin, Dmax, beta_t=5, beta_d=5):
        """
        Calculate user satisfaction based on throughput and delay.

        Parameters:
            throughput (float): The achieved throughput.
            delay (float): The achieved delay.
            Tmin (float): The minimum throughput requirement.
            Dmax (float): The maximum delay requirement.
            beta_t (float): Sigmoid steepness for throughput.
            beta_d (float): Sigmoid steepness for delay.

        Returns:
            float: The user satisfaction score.
        """
        sat_t = 1 / (1 + np.exp(np.clip(-beta_t * (throughput - Tmin), -500, 500)))
        sat_d = 1 / (1 + np.exp(np.clip(beta_d * (delay - Dmax), -500, 500)))
        return sat_t , sat_d
 

def main():
    # 创建卫星对象
    satellite1 = LEOSatellite(SatelliteID=1)
    satellite1.config_rbs(108)  # 配置卫星1有 108 个 RB
    satellite1.config_scs(15)  # 配置子载波间隔为 15 kHz

    # 创建用户终端
    UEs = {}
    num_apps_per_sat = 2  # 每个卫星的应用数量
    for i in range(1, num_apps_per_sat + 1):
        pos = Position(latitude=10.0 + i, longitude=20.0 + i)  # 随机生成位置
        ue = UserTerminal(pos=pos, common_satellites=[satellite1], id=i, flow_type="Video")
        UEs[i] = ue

    # 创建切片和应用
    slice1 = Slice(category="eMBB", satellite_id=1, number=1)
    slice2 = Slice(category="uRLLC", satellite_id=1, number=2)

    # 将应用添加到切片
    for i in range(1, num_apps_per_sat + 1):
        if(i % 2 == 0):
            slice1.add_app(UEs[i].app)  # eMBB 切片
        else:
            slice2.add_app(UEs[i].app)  # uRLLC 切片

    # 将切片添加到卫星
    satellite1._slices = [slice1, slice2]

    # 直接设置用户终端的连接
    for i in range(1, num_apps_per_sat + 1):
        UEs[i].connect_satellite = satellite1

    # 配置用户终端的 buffer，添加多个数据包
    for ue_id, ue in UEs.items():
        for _ in range(10):  # 每个用户终端添加 10 个数据包
            packet_size = np.random.choice([512, 1024, 2048])  # 随机选择数据包大小
            ue._push_packet(Packet(size=packet_size))

    # 创建调度器
    scheduler = MacScheduler(mac_slot_duration=10.0, slice_slot_duration=100.0, policy="ADMM")

    # 配置 SNR 数据
    num_rbs = 108  # 每个 UE 的 RB 数量
    snr_min, snr_max = 100, 120  # 信噪比范围（单位 dB）

    sat_snr_dict = {
        1: {  # 卫星1
            ue_id: {i: np.random.uniform(snr_min, snr_max, num_rbs) for i in range(10)}
            for ue_id in range(1, num_apps_per_sat + 1)
        },
    }

    # 配置 action_dict
    action_dict = {
        1: {  # 卫星1
            "embb_bd": 54,
            "urllc_bd": 54,
            "embb_power": 0.5,
            "urllc_power": 0.5,
            "ue_targets": list(range(1, num_apps_per_sat + 1)),
        },
    }

    # 调用 schedule 方法
    result = scheduler.schedule(leo_sats=[satellite1], action_dict=action_dict, UEs=UEs, sat_snr_dict=sat_snr_dict)

    # 检查调度是否成功
    if not result["success"]:
        print("Scheduling failed!")
        return

    # 调用 statistics_summary 方法
    app_results, slice_results = scheduler.statistics_summary(
        leo_sats=[satellite1],
        action_dict=action_dict,
        UEs=UEs,
        sat_snr_dict=sat_snr_dict,
        allocation=result["allocation"]
    )

    # 打印每个应用的统计信息
    print("App Statistics:")
    for app_id, stats in app_results.items():
        print(f"App {app_id}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # 打印每个切片的统计信息
    print("\nSlice Statistics:")
    for slice_key, arrival_rate in slice_results.items():
        print(f"Slice {slice_key}: Arrival Rate = {arrival_rate} bps")


if __name__ == "__main__":
    main()