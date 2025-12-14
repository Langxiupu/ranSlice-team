import datetime
import numpy as np
import gym
import uuid

# 为环境添加时间限制
# 该包装器会在指定的步数后自动结束环境
class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        # 显式暴露自定义属性
        if hasattr(env, "action_head_infos"):
            self.action_head_infos = env.action_head_infos
        if hasattr(env, "action_autoregressive_map"):
            self.action_autoregressive_map = env.action_autoregressive_map
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()
    
    def set_next_obs(self, obs):
        self.env.set_next_obs(obs)


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 检查动作空间的维度哪些是有限的
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        # 条件选择，如果动作空间的低值和高值是有限的，则使用它们，否则使用默认值
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        # 有限的动作空间将被转换为[-1, 1]范围
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)

# 离散动作空间
class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference

# 多头离散动作空间
class MultiHeadAction(gym.Wrapper):
    def __init__(self, env):
        """
        适配多头离散动作空间的Wrapper：
        1. 要求原始env自带action_head_infos和action_autoregressive_map属性
        2. 将每个动作头的离散动作转为One-Hot编码，保留原始环境的动作依赖关系
        
        Args:
            env: 原始环境（需满足以下条件）：
                - 动作空间为gym.spaces.Tuple，且每个元素为Discrete空间
                - 自带action_head_infos属性：list类型，每个元素为dict，含"name"（动作头名称）和"n"（动作数）
                - 自带action_autoregressive_map属性：list类型，长度与动作头数量一致，每个元素为依赖的前序动作头索引列表（-1表示无依赖）
        """
        # 校验原始环境的必备属性
        assert hasattr(env, "action_head_infos"), \
            "原始环境需自带action_head_infos属性（存储动作头元信息）"
        assert hasattr(env, "action_autoregressive_map"), \
            "原始环境需自带action_autoregressive_map属性（存储动作依赖关系）"
        
        # 校验动作空间类型与动作头数量匹配
        assert isinstance(env.action_space, gym.spaces.Tuple), \
            f"原始环境动作空间需为Tuple，当前为{type(env.action_space)}"
        assert len(env.action_space.spaces) == len(env.action_head_infos), \
            f"Tuple空间长度（{len(env.action_space.spaces)}）与动作头数量（{len(env.action_head_infos)}）不匹配"
        assert len(env.action_space.spaces) == len(env.action_autoregressive_map), \
            f"Tuple空间长度（{len(env.action_space.spaces)}）与依赖关系数量（{len(env.action_autoregressive_map)}）不匹配"
        
        # 校验每个动作头的元信息与Discrete空间一致性
        for idx, (space, head_info) in enumerate(zip(env.action_space.spaces, env.action_head_infos)):
            assert isinstance(space, gym.spaces.Discrete if env.action_head_infos[idx]["type"] == "categorical" else gym.spaces.Box), \
                f"动作头{idx}的空间需为Discrete，当前为{type(space)}"
            assert "name" in head_info and "out_dim" in head_info, \
                f"动作头{idx}的action_head_infos需包含'name'和'out_dim'字段"
            assert head_info["out_dim"] == space.n if hasattr(space, "n") else space.shape[0], \
                f"动作头{idx}：head_info['out_dim']（{head_info['out_dim']}）与空间维度（{space.n if hasattr(space, 'n') else space.shape[0]}）不匹配"
        
        # 校验依赖关系的合法性（依赖索引需为前序动作头或-1）
        for idx, dependencies in enumerate(env.action_autoregressive_map):
            for dep_idx in dependencies:
                if dep_idx != -1:
                    assert isinstance(dep_idx, int) and dep_idx < idx, \
                        f"动作头{idx}的依赖索引{dep_idx}不合法：需为-1或小于当前索引的整数"
        
        super().__init__(env)
        
        # 继承并暴露原始环境的动作头属性（对外可直接访问）
        self.action_head_infos = env.action_head_infos
        self.action_autoregressive_map = env.action_autoregressive_map
        
        # 构造多头One-Hot动作空间（Tuple包裹多个Box空间）
        self.action_space = gym.spaces.Tuple([
            gym.spaces.Discrete(head_info["out_dim"]) if head_info["type"] == "categorical" 
            else gym.spaces.Box(head_info["low"], head_info["high"], shape=(head_info["out_dim"],), dtype=np.float32)
            for head_info in env.action_head_infos
        ])
        # 标记动作空间属性（兼容上层逻辑判断）
        self.action_space.num_heads = len(env.action_head_infos)  # 标记动作头总数 
     
    def step(self, action):
        """
        处理多头动作(包含离散one-hot和连续动作)：
        1. 校验动作格式合法性
        2. 转换为原始环境需要的离散索引tuple
        3. 调用原始环境step方法
        
        Args:
            action (tuple): 多头One-Hot动作和连续动作，每个元素为对应动作头的np.ndarray（形状为(n,)）
                示例：(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, -0.2, 0.1, ...]))

        
        Returns:
            tuple: 原始环境step返回的(obs, reward, done, info)（Gymnasium为(obs, reward, terminated, truncated, info)）
        """       
        # 1. 基础格式校验
        assert isinstance(action, tuple), f"动作需为tuple类型，当前为{type(action)}"
        assert len(action) == len(self.action_head_infos), \
            f"动作长度（{len(action)}）与动作头数量（{len(self.action_head_infos)}）不匹配"
        
        # 2. 逐个动作头校验动作合法性并转换为离散索引
        actions = []
        for idx, (ac, head_info) in enumerate(zip(action, self.action_head_infos)):
            # 校验One-Hot向量形状
            assert isinstance(ac, np.ndarray) and ac.dtype in (np.float32, np.float64), \
                f"动作头{idx}：需为float32/float64的np.ndarray，当前为{type(ac)}（{ac.dtype}）"
            assert ac.shape == (head_info["out_dim"],), \
                f"动作头{idx}：动作形状需为({head_info['out_dim']},)，当前为{ac.shape}"
            
            # 校验One-Hot有效性（仅一个1，其余接近0，允许1e-5浮点误差）
            if head_info["type"] == "categorical":
                max_idx = np.argmax(ac).astype(int)
                reference = np.zeros_like(ac)
                reference[max_idx] = 1.0
                if not np.allclose(ac, reference, atol=1e-5):
                    raise ValueError(
                        f"动作头{idx}（{head_info['name']}）不合法：\n"
                        f"输入：{ac.round(4)}\n"
                        f"期望：{reference}"
                    )
                actions.append(max_idx)
            else:
                # 连续动作直接传递
                actions.append(ac)
        
        # 调用原始环境step（传入离散索引tuple）
        return self.env.step(tuple(actions)) 

    def reset(self):
        return self.env.reset()
    
    def set_next_obs(self, obs):
        """
        设置下一步的观测（仅用于调试）
        
        Args:
            obs (dict): 下一步的观测字典
        """
        self.env.set_next_obs(obs)
    
# 将奖励添加到观察空间
class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs

# 对于复合动作空间的环境，选择特定的动作
# 该包装器允许从复合动作空间中选择一个特定的动作键
class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        # 显式暴露自定义属性
        if hasattr(env, "action_head_infos"):
            self.action_head_infos = env.action_head_infos
        if hasattr(env, "action_autoregressive_map"):
            self.action_autoregressive_map = env.action_autoregressive_map
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])
    
    def set_next_obs(self, obs):
        self.env.set_next_obs(obs)

# 为每个环境实例生成唯一的UUID
class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 显式暴露自定义属性
        if hasattr(env, "action_head_infos"):
            self.action_head_infos = env.action_head_infos
        if hasattr(env, "action_autoregressive_map"):
            self.action_autoregressive_map = env.action_autoregressive_map
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()
    
    def set_next_obs(self, obs):
        self.env.set_next_obs(obs)
