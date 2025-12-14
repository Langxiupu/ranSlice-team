class CommApplication:
    def __init__(self, packet_size, data_rate, id=1):
        self.ID = id
        self._packet_size = packet_size
        self._data_rate = data_rate
        self._BW_u = None
        self._Delay_u = None

    def set_qos(self, bandwidth, delay):
        self._BW_u = bandwidth
        self._Delay_u = delay

    """
    获取当前app的带宽限制（单位：Mbps）
    """
    @property
    def BW_u(self):
        return self._BW_u

    @BW_u.setter
    def BW_u(self, value):
        self._BW_u = value

    """
    获取当前app的延迟限制（单位：ms）
    """
    @property
    def Delay_u(self):
        return self._Delay_u

    @Delay_u.setter
    def Delay_u(self, value):
        self._Delay_u = value

    """
    获取当前app的包大小（单位：bytes）
    """
    @property
    def packet_size(self):
        return self._packet_size

    # @packet_size.setter
    # def packet_size(self, value):
    #     self._packet_size = value

    """
    获取当前app的数据传输速率（单位：Mbps）
    """
    @property
    def data_rate(self):
        return self._data_rate

    # @data_rate.setter
    # def data_rate(self, value):
    #     self._data_rate = value

    def __repr__(self):
        return f"App(ID={self.ID}, BW_u={self.BW_u}, Delay_u={self.Delay_u}, packet_size={self.packet_size}, data_rate={self.data_rate})"