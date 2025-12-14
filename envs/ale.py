from collections import deque
import gym
import gym.spaces
import numpy as np
import cv2
from gym import spaces

class ALEEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 obs_type="rgb",  # 默认使用图像观测
                 action_repeat=4,
                 frame_size=(84, 84),
                 gray_scale=False,
                 noop_max=30,
                 life_done=False,
                 sticky=True,
                 actions="all",
                 seed=None):
        assert actions in ("all", "needed"), actions
        self._sticky = sticky  # 是否使用粘性动作
        self._obs_type = obs_type  # 观测类型（"rgb"或"ram"）
        # 初始化ALE环境
        self.env = gym.make(
            "ALE/Pong-v5", 
            obs_type=obs_type,  # 显式指定观测类型
            frameskip=1,        # 原始帧率控制
            repeat_action_probability=0.25 if sticky else 0.0,
            full_action_space=(actions == "all"),
        )

        # 正确设置种子
        if seed is not None:
            self.env.reset(seed=seed)
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)
        
        # 参数设置
        self.obs_type = obs_type
        self.action_repeat = action_repeat
        self.frame_size = frame_size # 添加通道维度
        self.gray_scale = gray_scale
        self.noop_max = noop_max
        self.life_done = life_done
        self.lives = self.env.unwrapped.ale.lives()  # 初始化生命值
        
        # 状态缓存
        self.frame_buffer = deque(maxlen=2)
        self.reset()

    @property
    def observation_space(self):
        if self._obs_type == "rgb":
            img_shape = self.frame_size + ((1,) if self.gray_scale else (3,))
            return spaces.Dict({
                "image": spaces.Box(0, 255, img_shape, np.uint8),
            })
        else:  # ram模式
            return spaces.Dict({
                "ram": spaces.Box(0, 255, (128,), np.uint8)  # Atari RAM为128字节
            })

    @property
    def action_space(self):
        space = self.env.action_space
        space.discrete = True
        return space

    def _process_frame(self, frame):
        # 返回frame的形状
        # print(f"Processing frame shape: {frame.shape} + {frame}")
        # 统一处理RGB和RAM观测
        if self._obs_type == "rgb":
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
            if self.gray_scale:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[..., None]
            return {"image": frame}  # 补零RAM
        else:
            return {"ram": frame.astype(np.uint8)}  # 直接返回RAM数据

    def step(self, action):
        """执行动作并返回字典格式的观测值"""
        total_reward = 0.0
        terminated = False
        truncated = False
        lost_life = False
    
        for _ in range(self.action_repeat):
            # 新版Gym API返回5个值
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
        
            # 生命值检测（确保info中有'lives'键）
            current_lives = info.get('lives', self.lives)  # 安全获取
            if self.life_done and current_lives < self.lives:
                lost_live = True
            self.lives = current_lives
        
            if terminated or truncated or lost_life:
                break
    
        # 统一返回格式
        processed = self._process_frame(obs)
        if self._obs_type == "rgb":
            obs_dict = {
                "image": processed["image"],
                "is_first": False,
                "is_terminal": terminated or lost_life,
                "is_last": truncated,
                "reward": total_reward
            }
        else:
            obs_dict = {
                "ram": processed["ram"],
                "is_first": False,
                "is_terminal": terminated or lost_life,
                "is_last": truncated,
                "reward": total_reward
            }
        return obs_dict, total_reward, terminated or lost_life, info

    def reset(self):
        """重置环境并返回字典形式的观测值（兼容DreamerV3的多模态输入）"""
        # 1. 获取原始观测
        obs, _ = self.env.reset()  # 假设是新版Gym（返回obs, info）
    
        # 2. 随机NOOP操作（保持原有逻辑）
        self.lives = self.env.unwrapped.ale.lives()
        for _ in range(np.random.randint(0, self.noop_max + 1)):
            obs, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated or (self.env.unwrapped.ale.lives() < self.lives):
                obs, _ = self.env.reset()
    
        # 统一返回字典格式
        if self._obs_type == "rgb":
            self.frame_buffer.clear()
            processed = self._process_frame(obs)
            for _ in range(2):
                self.frame_buffer.append(processed["image"])
            obs_dict = {
                "image": np.maximum(self.frame_buffer[-2], self.frame_buffer[-1]),
                "is_first": True,
                "is_terminal": False,
                "is_last": False
            }
        else:
            obs_dict = {
                "ram": self._process_frame(obs)["ram"],  # 原始RAM数据
                "is_first": True,
                "is_terminal": False,
                "is_last": False
            }
        return obs_dict

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()