import pickle

# 1. 读取 pkl 文件
with open("sat-data/visibles_t_u-46users.pkl", "rb") as f:
    data = pickle.load(f)

"""
data: list  2000 代表2000个时隙，共计200秒，一个时隙0.1秒
data[0]: dict 代表46个用户，key代表用户ID，从0-45，value代表可见卫星集合
data[0][0]: 第0个用户的可见卫星集合，dict,key代表卫星ID，value代表VisibleSat类
data[0][0][38]: 代表第0时刻和用户0可见的卫星38，卫星类有5个属性：
sat_id sat_name altitude azimuth range
"""


def get_common_visible_sats(data, start_idx, end_idx, num_users=46, slot_len=0.1):
    """
    在时间区间 [start_idx, end_idx] 内，
    找出 46 个用户在所有时刻均可见的公共卫星集合。

    :param data:   列表，长度约 2000
    :param start_idx: 起始时隙索引（包含）
    :param end_idx:   终止时隙索引（包含）
    :param num_users: 用户数量，默认 46
    :param slot_len:  每个时隙的秒数，默认 0.1s
    :return:
        common_sats:  set，公共可见卫星 ID 集合
        t_start_sec:  起始时间（秒）
        t_end_sec:    终止时间（秒）
    """
    if start_idx < 0 or end_idx >= len(data) or start_idx > end_idx:
        raise ValueError("时间下标不合法")

    common_sats = None  # 最终在整个区间内都可见的卫星集合

    # 遍历每个时刻
    for t in range(start_idx, end_idx + 1):
        slot = data[t]  # dict: user_id -> {sat_id: VisibleSat}

        # 先在当前时刻内，求所有用户的公共可见卫星
        common_at_t = None
        for user_id in range(num_users):
            # 当前用户在时刻 t 可见的卫星 ID 集合
            sat_ids = set(slot[user_id].keys())

            if common_at_t is None:
                common_at_t = sat_ids
            else:
                common_at_t &= sat_ids

            # 如果这一时刻已经没有公共卫星了，提前退出
            if not common_at_t:
                break

        # 第一个时刻，初始化整体公共集合
        if common_sats is None:
            common_sats = common_at_t
        else:
            # 和之前时刻的公共集合做交集，保证整个时间段都可见
            common_sats &= common_at_t

        # 如果整个区间已经没有公共卫星了，也可以提前退出
        if not common_sats:
            break

    t_start_sec = start_idx * slot_len
    t_end_sec = end_idx * slot_len
    return common_sats, t_start_sec, t_end_sec


# ================= 使用示例 =================
if __name__ == "__main__":
    # 比如你想试一下从第 100 个时隙到第 300 个时隙
    x = 0
    y = 1000

    common_sats, t_start, t_end = get_common_visible_sats(data, x, y)

    print(f"时间区间：index [{x}, {y}]  ->  秒 [{t_start:.1f}s, {t_end:.1f}s]")
    print("公共可见卫星ID集合：", sorted(common_sats) if common_sats else set())