from numbers import Integral
import gym
import numpy as np
import math


class DummyMultiHeadEnv(gym.Env):
    def __init__(self):
        # 观测空间：旧版 Gym 支持 Dict 空间
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(-1, 1, shape=(10,)),
            "achieved_goal": gym.spaces.Box(-1, 1, shape=(2,3)),
        })
        # 动作空间：旧版 Gym 支持 Tuple 空间
        self._action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(1),
            gym.spaces.Discrete(4),
            gym.spaces.Discrete(4),
            gym.spaces.Box(low=0, high=1, shape=(2,)),
        ))
        self.reward_range = (-math.inf, math.inf)
        self.metadata = None

        # 动作头元信息（保持不变）
        self.head_infos = [
            {"name": "1", "type": "categorical", "out_dim": 1},
            {"name": "2", "type": "categorical", "out_dim": 4},
            {"name": "3", "type": "categorical", "out_dim": 4},
            {"name": "4", "type": "continuous", "out_dim": 2, "low": 0, "high": 1},
        ]
        self.autoregressive_maps = [
            [-1],
            [-1],
            [1],
            [-1]
        ]
        self.action_type_masks = [
            [1, 1, 1],
        ]

    def _get_state(self, is_first=False, is_terminal=False):
        # 生成观测（保持不变）
        observation = np.clip(np.random.randn(10), -1, 1)
        achieved_goal_item = np.clip(np.random.randn(3), -1, 1)
        achieved_goal = np.tile(achieved_goal_item, (2,1))
        return {"observation": observation, "achieved_goal": achieved_goal, "is_first": is_first, "is_terminal": is_terminal}

    def get_observation_dict_info(self):
        # 保持不变：返回观测空间维度
        return {key: space.shape for key, space in self.observation_space.spaces.items()}

    def get_available_actions(self):
        # 保持不变：生成可用动作掩码
        avail_acs_all = []
        for i in range(len(self.head_infos)):
            ac_dim = self.head_infos[i]["out_dim"]
            avail_acs = np.random.rand(ac_dim) > 0.5
            ind = int(np.random.randint(0, ac_dim))
            avail_acs[ind] = 1.0
            avail_acs_all.append(avail_acs.astype(float))
        return avail_acs_all

    def get_observation_shapes(self):
        # 保持不变：返回观测空间形状信息
        sub_shapes = {}
        total_dim = 0
        for key, space in self.observation_space.spaces.items():
            # 判断每个space是连续还是离散
            if isinstance(space, gym.spaces.Box):
                sub_shapes[key] = space.shape
                total_dim += np.prod(space.shape)
            elif isinstance(space, gym.spaces.Discrete):
                sub_shapes[key] = space.n
                total_dim += space.n
        total_shape = (total_dim,)
        return total_shape  # 注：原代码漏了返回 sub_shapes，如需可改为 return sub_shapes, total_shape

    def reset(self, seed=None):
        # 旧版 Gym 接口：可选支持 seed，仅返回观测
        if seed is not None:
            self.seed(seed)
        return self._get_state(is_first=True, is_terminal=False)

    def step(self, action):
        # 修正动作校验逻辑，兼容 int（索引）和 one-hot（ndarray）
        for i, ac in enumerate(action):
            ac_dim = self.head_infos[i]["out_dim"]
            if self.head_infos[i]["type"] == "categorical":
                if isinstance(ac, Integral):
                    assert ac < ac_dim, f"动作头{i}：索引{ac}超出维度{ac_dim}"
                elif isinstance(ac, np.ndarray):
                    idx = np.argmax(ac)
                    assert idx < ac_dim, f"动作头{i}：one-hot 索引{idx}超出维度{ac_dim}"
                else:
                    raise TypeError(f"动作头{i}：支持 int 或 np.ndarray，当前类型{type(ac)}")
            elif self.head_infos[i]["type"] == "continuous":
                if isinstance(ac, np.ndarray):
                    assert ac.shape == (ac_dim,), f"动作头{i}：连续动作形状需为({ac_dim},)，当前为{ac.shape}"
                else:
                    raise TypeError(f"动作头{i}：连续动作需为 np.ndarray，当前类型{type(ac)}")

        # 旧版 Gym step 返回：(obs, reward, done, info) 4元组
        reward = np.random.rand()
        done = np.random.rand() < 0.1  # 旧版用 done 统一表示终止
        if done:
            state = self._get_state(is_first=False, is_terminal=True)
        else :
            state = self._get_state(is_first=False, is_terminal=False)
        info = {"available_actions": self.get_available_actions()}
        return state, reward, done, info

    def seed(self, seed):
        # 保持不变：设置随机种子
        np.random.seed(seed)
    
    @property
    def action_space(self):
        # 保持不变：动作空间属性
        return self._action_space
    
    @property
    def action_head_infos(self):
        # 保持不变：动作头元信息（供包装器使用）
        return self.head_infos
    
    @property
    def action_autoregressive_map(self):
        # 保持不变：动作依赖关系（供包装器使用）
        return self.autoregressive_maps