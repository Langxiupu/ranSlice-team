from commModule.apps import CommApplication


class Slice:
    """
    参数：
        category:切片的类型,eMBB是带宽密集型,uRLLC是延迟敏感性。
        satellite_id:卫星的ID
        number:当前卫星下该切片的序号
    """
    def __init__(self, category='eMBB', satellite_id=1, number=1):
        # 如果category是1，代表带宽密集型（典型业务：Vehicle-to-X）；如果是2，则是延迟敏感性（典型业务：Video conf）
        self._category = category
        self.config_service(category, satellite_id, number)
        self._BW = 0
        self.apps = {}  # 存储CommApplication对象的字典

    def typical_service(self, BW, Delay):
        self._BW_s = BW  # 单位 Mbps
        self._Delay_s = Delay  # 单位 ms

    """
    配置当前切片的类型与典型服务
    参数：
        category: 切片的类型，"eMBB"代表带宽密集型，"uRLLC"代表延迟敏感型
        satellite_id: 卫星的ID
        number: 当前卫星下该切片的序号
    """
    def config_service(self, category, satellite_id, number):
        # 统一转为小写处理，增强容错性（如允许"Embb"、"URLLC"等输入）
        category_lower = category.lower()

        if category_lower == "embb":
            # 带宽密集型服务（eMBB）
            self.typical_service(BW=2, Delay=150)
            self._slice_id = f"{number}-{satellite_id}"+"-eMBB"
            # 所属卫星ID
            self._satellite_id = satellite_id
        elif category_lower == "urllc":
            # 延迟敏感型服务（uRLLC）
            self.typical_service(BW=0.2, Delay=50)
            self._slice_id = f"{number}-{satellite_id}"+"-uRLLC"
            # 所属卫星ID
            self._satellite_id = satellite_id
        else:
            # 处理无效类型
            raise ValueError(f"不支持的切片类型: {category}，请使用 'eMBB' 或 'uRLLC'")

    """
    向当前切片slice的app字典中添加app
    参数：
        app:具体的应用
    """
    def add_app(self, app):
        if isinstance(app, CommApplication):
            # 以应用的ID作为键，CommApplication对象作为值存储
            self.apps[app.ID] = app
        else:
            raise TypeError("app must be an instance of CommApplication")

    """
    向当前切片slice的app字典中删除app
    参数：
        id:具体的应用的ID
    """
    def remove_app(self, id):
        if id in self.apps:
            del self.apps[id]
        else:
            raise ValueError(f"app with ID {id} not found in the slice")

    """
    获取当前切片的类型
    """
    @property
    def category(self):
        return self._category

    # @category.setter
    # def category(self, value):
    #     self._category = value

    """
    获取当前切片典型服务的带宽限制（单位：Mbps）
    """
    @property
    def BW_s(self):
        return self._BW_s

    @BW_s.setter
    def BW_s(self, value):
        self._BW_s = value

    """
    获取当前切片典型服务的延迟限制（单位：ms）
    """
    @property
    def Delay_s(self):
        return self._Delay_s

    @Delay_s.setter
    def Delay_s(self, value):
        self._Delay_s = value

    """
    获取当前切片的带宽限制（单位：Mbps）
    """
    @property
    def BW(self):
        return self._BW

    @BW.setter
    def BW(self, value):
        self._BW = value

    """
    获取当前切片的ID
    """
    @property
    def slice_id(self):
        return self._slice_id

    @slice_id.setter
    def slice_id(self, value):
        self._slice_id = value

    def calculate_distance(self, app):
        # 通过欧几里得距离计算典型服务与具体app间的距离
        BW_u = app.BW_u
        Delay_u = app.Delay_u
        return ((self._BW_s - BW_u) ** 2 + (self._Delay_s - Delay_u) ** 2) ** 0.5

    def __repr__(self):
        return f"Slice(category={self._category}, BW_s={self._BW_s}, Delay_s={self._Delay_s}, BW={self._BW}, slice_id={self._slice_id})"
