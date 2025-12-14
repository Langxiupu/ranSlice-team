import json
import os

def _convert_tuples(obj):
    """递归将所有元组转换为带标记的字典"""
    if isinstance(obj, tuple):
        return {"__tuple__": True, "values": [_convert_tuples(item) for item in obj]}
    elif isinstance(obj, list):
        return [_convert_tuples(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _convert_tuples(v) for k, v in obj.items()}
    else:
        return obj

def _restore_tuples(obj):
    """递归将带标记的字典还原为元组"""
    if isinstance(obj, dict):
        if "__tuple__" in obj:
            return tuple([_restore_tuples(item) for item in obj["values"]])
        return {k: _restore_tuples(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_restore_tuples(item) for item in obj]
    else:
        return obj

def save_env_config(env_config, file_path="env_config.json"):
    """将环境配置字典保存为JSON文件"""
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    # 手动转换所有元组后再序列化
    converted_config = _convert_tuples(env_config)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(converted_config, f, ensure_ascii=False, indent=2)
    print(f"配置已保存至: {os.path.abspath(file_path)}")

def load_env_config(file_path="env_config.json"):
    """从JSON文件读取配置"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件不存在: {os.path.abspath(file_path)}")
    with open(file_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    # 手动还原所有元组
    restored_config = _restore_tuples(config)
    print(f"配置已从 {os.path.abspath(file_path)} 加载")
    return restored_config


if __name__ == "__main__":
    env_config = {
        "visibles_t_u": "sat-data/visibles_t_u-46users.pkl",
        "ue_position": "data/random_users46.csv",
        "traffic_wide": "sat-data/traffic-wide-13pm.pkl",
        "sats_bd_rbs": "sat-data/sats-bd-rb.csv",
        "apps_weights": "sat-data/apps-weights.csv",
        "app_flow_types": "sat-data/app_flow_types.csv",
        "slice_slots": 1000,  # 切片时隙数量
        "common_sats": [1143, 1644, 1666, 3743],
        "short_slot_interval": 0.01, # 10ms
        "lambda": 0.5, # 0.5
        "slice_interval": 0.1, # 100ms
        "schedulers": [
            {
                "policy": "RR",
                "params": (20e6, 15e3)  # 元组
            }
        ],
        "flow_gen_method": {
            "type": "on-off",
            "params": (0.5,)  # 单元素元组
        }
    }

    save_env_config(env_config)
    loaded_config = load_env_config()

    # 验证元组类型是否还原
    print(f"调度器参数类型: {type(loaded_config['schedulers'][0]['params'])}")
    print(f"流生成参数类型: {type(loaded_config['flow_gen_method']['params'])}")
