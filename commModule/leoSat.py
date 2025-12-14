from commModule.slice import Slice
class LEOSatellite:
    def __init__(self, SatelliteID=1):
        self.ID = SatelliteID
        # 目前暂定每个卫星中有两种slice各一个，用一个slice列表存储这两个

        self._slices = []
        self._BW = 0  # 卫星的带宽，单位为MHz
        self._rbs = 0 # 卫星分得的rbs数量
        self._scs = 15  # 子载波间隔，单位为kHz，默认15kHz
        self.max_tx_power = 30 # 最大通信功率，单位：dbm
        self.add_slice(1, category="eMBB")  # 添加一个带宽密集型slice
        self.add_slice(1, category="uRLLC")  # 添加一个延迟敏感

    def config_mac(self, slice, macscheduler, operation="add"):
        # 参数是一个slice和一个mac调度器对象，mac调度器对象将管理这个slice
        if slice in self._slices:
            if operation == "add":
                macscheduler.add_slice(slice)  # 将slice添加到mac调度器中
            elif operation == "remove":
                macscheduler.remove_slice(slice)  # 将slice从mac调度器中移除
            else:
                raise ValueError("Invalid operation. Use 'add' or 'remove'.")
        else:
            raise ValueError("The slice is not part of this satellite's slices.")

    # 这个方法会将该颗卫星的所有切片都加入或删除到指定的mac调度器中
    def config_mac_scheduler(self, macscheduler, operation="add"):
        if operation == "add":
            for slice in self._slices:
                macscheduler.add_slice(slice)  # 将每个slice添加到mac调度器中
        elif operation == "remove":
            for slice in self._slices:
                macscheduler.remove_slice(slice)  # 将每个slice从mac调度器中移除
        else:
            raise ValueError("Invalid operation. Use 'add' or'remove'.")

    def config_BW(self, bandwidth):  # 配置卫星的带宽(单位为MHz)
        self._BW = bandwidth

    def config_rbs(self, rbs):  # 配置卫星的rbs数量
        self._rbs = rbs
    
    def config_scs(self, scs):  # 配置卫星的子载波间隔(单位为kHz)
        self._scs = scs

    def add_slice(self, num_slice, category="eMBB"):
        # num_slice是slice的数量，category是slice的类型，"eMBB"代表带宽密集型，"uRLLC"代表延迟敏感型
        for i in range(num_slice):
            slice = Slice(category=category, satellite_id=self.ID, number=i + 1)
            self._slices.append(slice)

    """
    这个config_slice方法应具备从slice中添加和删除应用的功能
    根据通过欧几里得距离计算典型服务与具体app间的距离来选择对应slice
    参数:
        app: 具体的应用
        operation: 具体采用的操作，"add"是添加应用到slice中，"remove"是从slice中删除应用
    """
    def config_slice(self, app, operation="add"):  # 默认操作设为添加
        min_distance = float('inf')
        closest_slice = None
        for slice in self._slices:
            distance = slice.calculate_distance(app)
            if distance < min_distance:
                min_distance = distance
                closest_slice = slice
        # 根据操作类型执行对应逻辑
        if operation == "add":
            closest_slice.add_app(app)
        elif operation == "remove":
            closest_slice.remove_app(app.ID)
        else:
            # 处理无效操作的情况
            raise ValueError(f"不支持的操作: {operation}，请使用 'add' 或 'remove'")

    """
    获取当前卫星下的所有slice列表
    """
    @property
    def slices(self):
        return self._slices

    """
    获取当前卫星的带宽
    """
    @property
    def BW(self):
        return self._BW

    """
    获取当前卫星的rbs
    """
    @property
    def rbs(self):
        return self._rbs
    """
    获取当前卫星的子载波间隔
    """
    @property
    def scs(self):
        return self._scs
    """
    最大通信功率
    """
    @property
    def p_max(self):
        return self.max_tx_power
    def __repr__(self):
        return f"LEOSatellite(ID={self.ID})"

