import time
from scipy.stats import pareto
from queue import Queue
from commModule.apps import CommApplication


class Position:
    # 位置类，用于表示用户终端的位置，两个属性经纬度来表示
    def __init__(self, latitude=0.0, longitude=0.0):
        self._latitude = latitude
        self._longitude = longitude
    """
    获取当前位置的纬度
    """
    @property
    def latitude(self):
        return self._latitude

    # @latitude.setter
    # def latitude(self, value):
    #     self._latitude = value

    """
    获取当前位置的经度
    """
    @property
    def longitude(self):
        return self._longitude

    # @longitude.setter
    # def longitude(self, value):
    #     self._longitude = value



class Distance:
    # 用于表示用户终端与卫星之间的距离，包括高度角 (Altitude)，方位角 (Azimuth)，距离 (Range)。
    def __init__(self, altitude=0.0, azimuth=0.0, Range=0.0):
        self.altitude = altitude
        self.azimuth = azimuth
        self.range = Range  # 距离单位为km

    # 重新定义比较大小，通过距离的range属性进行比较
    def __lt__(self, other):
        if isinstance(other, Distance):
            return self.range < other.range
        return NotImplemented

    def __repr__(self):
        return f"Distance(altitude={self.altitude}, azimuth={self.azimuth}, range={self.range})"


# 不同业务类型的流传输
class Flow:
    def __init__(self, flow_type=1):
        self.flow_type = flow_type  # 流类型，例如视频流、语音流等
        self.config_flow()

    def config_flow(self):
        # 根据流类型配置流的对应参数
        # 类型1是语音流
        if self.flow_type == 1:
            self._QCI = 20  # QCI值
            self._data_rate = 0.2  # 数据速率(单位为Mbps)
            self._latency = 100  # 延迟
            self._packet_size = 128  # 包大小（单位为Bytes）
        # 类型2是车联网流
        elif self.flow_type == 2:
            self._QCI = 40
            self._data_rate = 0.2
            self._latency = 50
            self._packet_size = 256  # 包大小（单位为Bytes）
        # 类型3是视频流
        elif self.flow_type == 3:
            self._QCI = 40
            self._data_rate = 2 # 2 Mbps
            self._latency = 150
            self._packet_size = 1024  # 包大小（单位为Bytes）
        # 类型4是VR/AR流
        elif self.flow_type == 4:
            self._QCI = 68
            self._data_rate = 10 # 10 Mbps
            self._latency = 30
            self._packet_size = 1500  # 包大小（单位为Bytes）
        else:
            raise ValueError("Unsupported flow type. Supported types are 1, 2, 3, and 4.")

    def __repr__(self):
        return (f"Flow(flow_type={self.flow_type}, QCI={self.QCI}, "
                f"data_rate={self.data_rate}, latency={self.latency}, "
                f"packet_size={self.packet_size})")

    def get_flow_type(self):
        if self.flow_type == 1:
            return "Voice"
        elif self.flow_type == 2:
            return "V2X"
        elif self.flow_type == 3:
            return "Video"
        elif self.flow_type == 4:
            return "VR/AR"
        else:
            return "Unknown"

    """
    获取当前流的QCI值
    """
    @property
    def QCI(self):
        return self._QCI

    # @QCI.setter
    # def QCI(self, value):
    #     self.QCI = value

    """
    获取当前流的数据传输速率（单位：Mbps）
    """
    @property
    def data_rate(self):
        return self._data_rate

    # @data_rate.setter
    # def data_rate(self, value):
    #     self._data_rate = value

    """
    获取当前流的延迟（单位：ms）
    """
    @property
    def latency(self):
        return self._latency

    # @latency.setter
    # def latency(self, value):
    #     self._latency = value

    """
    获取当前流的包大小（单位：Bytes）
    """
    @property
    def packet_size(self):
        return self._packet_size

    # @packet_size.setter
    # def packet_size(self, value):
    #     self._packet_size = value



# 数据包类,主要参数为包大小以及包的创建时间
class Packet:
    def __init__(self, size=1000):
        self.size = size  # 包大小(单位为Byte)
        self.current_size = 0  # 当前包大小(单位为Byte)
        self.transmission_ratio = 0.0  # 当前数据包传输比例(0-1)
        self.gen_time = 0

        # 传输相关变量
        self.start_transmission_time = None  # 开始传输的时间 (ms)
        self.transmitted_size = 0  # 已经传输的大小 (B)

    def get_size(self):
        return self.size

    def get_current_size(self):
        return self.current_size

    """
    更新当前生成数据包大小
    参数：
        data:生成的数据大小，单位为B

    返回：
        remain_data:盈余的数据大小，单位为B
    """
    def update_current_size(self, data):
        if data < 0:
            raise ValueError("Data size must be non-negative.")
        if self.current_size + data > self.size:
            remain_data = self.current_size + data - self.size
            self.current_size = self.size
            return remain_data
        else:
            self.current_size += data
            return 0

    """
    更新数据包传输比例
    参数：
        data:传输的数据大小，单位为B
    返回：
        remain_data:剩余未传输的数据大小，单位为B
    """
    def update_transmission_ratio(self, data):
        if data < 0:
            raise ValueError("Data size must be non-negative.")
        if self.transmission_ratio + (data+0.0) / self.size > 1:
            remain_data = data - (1 - self.transmission_ratio) * self.size
            self.transmission_ratio = 1.0
            return remain_data
        else:
            self.transmission_ratio += (data + 0.0) / self.size
            return 0

    def set_gen_time(self, time_val=0):
        self.gen_time = time_val


class UserTerminal:
    """
    参数：
        pos: 用户终端的位置，类型为Position
        common_satellites: 所有用户终端共享的卫星列表，类型为LEOSatellite的列表
        id: 用户终端的唯一标识符，默认为1
        flow_type: 流类型，默认为Voice（语音流）
    """
    def __init__(self, pos, common_satellites, id=1, flow_type="Voice"):
        self.ID = id
        self.flow_type = flow_type  # 流类型
        self.buffer = Queue()  # 存储生成的数据包的缓冲队列
        self.connect_satellite = None  # 连接到当前UE的卫星
        self.app = None  # 与该UE绑定的应用
        self.flow_service = None  # 该UE对应的流服务
        self.common_satellites = common_satellites  # 所有UE的共同卫星字典, *e.g [{LEOSatellite(1)},{LEOSatellite(1644)}]
        self.common_distances = None  # 所有UE的共同卫星到该UE的距离字典, *e.g [1:{Distance},1644:{Distance}]
        self.current_time = 0  # 当前时间
        self.current_gen_pkt = None  # 当前正在生成的数据包
        self.delay_pkt = []  # 记录每个包的传输延迟
        self.position = pos  # 用户终端的位置
        self.config_apps()  # 配置应用
        self.active = False  # 用户终端是否活跃
        ...

    """
    配置用户终端的距离和卫星
    参数:
        distances: 卫星与用户终端之间的距离，类型为Distance的字典，键为卫星ID，值为对应的Distance对象
    """
    def config_distances(self, distances):
        if isinstance(distances, dict):
            self.common_distances = distances
        else:
            raise TypeError("距离必须是一个字典，键为卫星ID，值为对应的Distance对象。")

    """
    为用户终端配置APP
    """
    def config_apps(self):
        if self.app is None:
            self.config_flow()
            self.app = CommApplication(self.flow_service.packet_size, self.flow_service.data_rate * 1.2, id=self.ID)
            self.app.set_qos((self.flow_service.data_rate * 1.2), self.flow_service.latency)

    def config_flow(self):
        # 定义字符串到数字的映射关系（不区分大小写）
        flow_type_map = {
            'voice': 1,
            'v2x': 2,
            'video': 3,
            'vr/ar': 4
        }
        if self.flow_service is None:
            # 将输入转换为小写后查找映射，若未找到匹配项可根据需求处理（此处示例为抛出异常）
            lower_flow_type = self.flow_type.lower()
            if lower_flow_type not in flow_type_map:
                raise ValueError(f"不支持的flow_type: {self.flow_type}，支持的类型为Voice、V2X、Video、VR/AR")
            # 获取对应的数字类型并创建Flow实例
            numeric_flow_type = flow_type_map[lower_flow_type]
            self.flow_service = Flow(flow_type=numeric_flow_type)
    
    def get_flow_data_rate(self): # 获取当前ue的流服务的数据传输速率
        if self.flow_service is None:
            return 0
        else:
            return self.flow_service.data_rate

    def config_flow_method(self, flow_method="on-off"):
        # 定义字符串到数字的映射关系
        method_map = {
            "on-off": 0,
            "pareto": 1
        }
        # 将输入转换为小写（可选，增强容错性）
        flow_method_lower = flow_method.lower()
        # 检查输入是否有效
        if flow_method_lower not in method_map:
            raise ValueError(f"不支持的流方法: {flow_method}，支持的方法为'on-off'和'pareto'")
        # 设置对应的数字值
        self.flow_method = method_map[flow_method_lower]

    def connect_sat(self):
        #  与卫星建立连接的方式是通过距离来选择最近的卫星连接
        #  根据common_satellites中的卫星id从common_distances中找到对应的最短距离卫星对象
        if self.common_distances is None or self.common_satellites is None:
            raise ValueError("距离字典或卫星字典为空。")
        min_distance = Distance(altitude=float('inf'), azimuth=float('inf'), Range=float('inf'))
        min_satellite = None
        self.active = True
        for satellite in self.common_satellites:
            distance = self.common_distances[satellite.ID]
            if distance < min_distance:
                min_distance = distance
                min_satellite = satellite
        if self.connect_satellite is None:
            self.connect_satellite = min_satellite
            self.connect_satellite.config_slice(self.app, "add")  # 将应用添加到新的切片中
        elif self.connect_satellite != min_satellite:
            self.connect_satellite.config_slice(self.app, "remove")  # 首先将应用从原来的切片中删除
            self.connect_satellite = min_satellite
            self.connect_satellite.config_slice(self.app, "add")  # 然后将应用添加到新的切片中
    
    def connect_specific_sat(self, satellite):
        # 连接到指定的卫星
        if self.connect_satellite is None:
            self.connect_satellite = satellite
            self.connect_satellite.config_slice(self.app, "add")  # 将应用添加到新的切片中
            self.active = True
        elif self.connect_satellite != satellite:
            self.connect_satellite.config_slice(self.app, "remove")  # 首先将应用从原来的切片中删除
            self.connect_satellite = satellite
            self.connect_satellite.config_slice(self.app, "add")  # 然后将应用添加到新的切片中
    
    # 返回当前正在连接的卫星的id
    def get_connect_sat_id(self):
        if self.connect_satellite is not None:
            return self.connect_satellite.ID
        else:
            return None

    def disconnect_sat(self):
        # 断开与卫星的连接，将应用从切片中删除
        self.active = False
        if self.connect_satellite is not None:
            self.connect_satellite.config_slice(self.app, "remove")  # 将应用从原来的切片中删除
            self.connect_satellite = None

    def _push_packet(self, packet):
        # 将生成的数据包推入到buffer中
        if isinstance(packet, Packet):
            self.buffer.put(packet)
        else:
            raise TypeError("数据包必须是Packet类型。")

    def _reset_buffer(self):
        # 清空buffer
        self.buffer = Queue()
    
    def queue_length(self):
        return self.buffer.qsize()

    """
    根据终端指定的流生成方式配置相关参数
    参数：
        *args: 流生成方式所需的参数，根据不同的流生成方式可能有不同的参数
    注意：
        on-off模型：只需要1个参数（offload）
        重尾流分布：需要2个参数（alpha, min_flow）
    """
    def config_flow_params(self, args):
        # on-off模型：只需要1个参数（offload）
        if self.flow_method == 0:
            # 检查参数数量，确保只传入1个参数
            if len(args) != 1:
                raise ValueError("on-off模型需要且仅需要1个参数（offload）")
            offload = args[0]
            # 计算发送一个包所需的时间
            sendf = (self.flow_service.packet_size * 8) / (self.app.data_rate * 1000000)  # 单位：s
            self.onTime = sendf  # on阶段的时间
            self.offTime = sendf * (1 - offload) / offload  # off阶段的时间
            self.onCurrentTime = 0  # 当前on阶段已持续时间
            self.offCurrentTime = 0  # 当前off阶段已持续时间
        # 重尾流分布：需要2个参数（alpha, min_flow）
        elif self.flow_method == 1:
            # 检查参数数量，确保传入2个参数
            if len(args) != 2:
                raise ValueError("重尾流模型需要且仅需要2个参数（alpha, min_flow）")
            alpha, min_flow = args  # 按顺序解析参数
            self.pareto_flow_volume = self._generate_pareto_value(alpha, min_flow)
        else:
            raise ValueError("不支持的流生成方法。支持的方法为0（on-off）和1（Pareto）")

    """
    根据终端指定的流生成方式生成数据包并存入buffer中，一共两种，一种是连续不断，另一种是on-off模式
    参数:
        time_interval: 每一次short slot调度的时间间隔，默认为0.01秒
    """
    def generate_flow(self, time_interval=0.01):
        if (self.flow_method == 0):
            # on-off模型
            # 第一步，计算出在一个time_interval调度下分别在on和off两个阶段的总时间
            on_time_total = 0.0
            off_time_total = 0.0
            time_total = time_interval
            while (time_total > 0):
                if self.onCurrentTime < self.onTime:
                    # 在on阶段
                    if self.onCurrentTime + time_total < self.onTime:
                        on_time_total += time_total
                        self.onCurrentTime += time_total
                        time_total = 0
                    else:
                        on_time_total += self.onTime - self.onCurrentTime
                        time_total -= (self.onTime - self.onCurrentTime)
                        self.onCurrentTime = self.onTime
                        self.offCurrentTime = 0
                else:
                    # 在off阶段
                    if self.offCurrentTime + time_total < self.offTime:
                        off_time_total += time_total
                        self.offCurrentTime += time_total
                        time_total = 0
                    else:
                        off_time_total += self.offTime - self.offCurrentTime
                        time_total -= (self.offTime - self.offCurrentTime)
                        self.offCurrentTime = self.offTime
                        self.onCurrentTime = 0
            # 第二步，根据on阶段的时间生成数据包
            # 首先得到on阶段的时间对应的生成数据大小
            data_to_generate = on_time_total * (self.app.data_rate *1000000) / 8  # 转换为Bytes
            total_data = data_to_generate  # 记录总的数据量，用于计算较为精确的包生成时间
            while on_time_total > 0 and data_to_generate > 0:
                if self.current_gen_pkt is None:
                    # 生成新的数据包
                    self.current_gen_pkt = Packet(size=self.flow_service.packet_size)
                data_to_generate = self.current_gen_pkt.update_current_size(data_to_generate)
                if self.current_gen_pkt.current_size == self.current_gen_pkt.size:
                    # 如果数据包已经满了，则将其推入到buffer中
                    self.current_gen_pkt.set_gen_time(self.current_time + (time_interval * (total_data - data_to_generate) / total_data))
                    self._push_packet(self.current_gen_pkt)
                    self.current_gen_pkt = None
        elif (self.flow_method == 1):
            # 重尾流分布不间断生成
            data_to_generate = min(self.pareto_flow_volume * 1000000, time_interval * (self.app.data_rate * 1000000) / 8)
            total_data = data_to_generate  # 记录总的数据量，用于计算较为精确的包生成时间
            while data_to_generate > 0:
                if self.current_gen_pkt is None:
                    # 生成新的数据包
                    self.current_gen_pkt = Packet(size=self.flow_service.packet_size)
                data_to_generate = self.current_gen_pkt.update_current_size(data_to_generate)
                if self.current_gen_pkt.current_size == self.current_gen_pkt.size:
                    # 如果数据包已经满了，则将其推入到buffer中
                    self.current_gen_pkt.set_gen_time(self.current_time+(time_interval*(total_data-data_to_generate)/total_data))
                    self._push_packet(self.current_gen_pkt)
                    self.current_gen_pkt = None
            self.pareto_flow_volume -= time_interval * self.flow_service.data_rate / 8  # 更新剩余的重尾流分布数据量
        else:
            raise ValueError("Unsupported flow generation method. Supported methods are 0 (on-off) and 1 (Pareto).")
        self.current_time += time_interval

    def _generate_pareto_value(self, alpha, x_min):
        """
        生成一个符合帕累托分布的随机值
        参数:
            alpha: 形状参数（α > 0，越小重尾越明显）
            x_min: 最小值阈值（x ≥ x_min）
        返回:
            随机生成的值（≥ x_min）
        """
        return pareto(b=alpha, scale=x_min).rvs(size=1)[0]

    def __repr__(self):
        return f"UserTerminal(ID={self.ID}, flow_type={self.flow_type}, position={self.position}"

    #reset函数，用于重置用户终端的状态
    def reset(self):
        self.disconnect_sat()  # 断开与卫星的连接
        self._reset_buffer()  # 清空buffer
        self.current_gen_pkt = None  # 清空当前正在生成的数据包
        self.delay_pkt = []  # 清空延迟记录列表
        self.current_time = 0  # 重置当前时间