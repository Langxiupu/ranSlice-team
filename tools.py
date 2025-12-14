import datetime
import collections
import io
import os
import json
import pathlib
import re
import time
import random

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter


to_np = lambda x: x.detach().cpu().numpy()

# 张量以log形式存储
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

# 张量以指数形式存储
def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

# 智能管理模型参数的梯度计算状态
class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)

# 精确测量CUDA操作的执行时间
class TimeRecording:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Logger:
    def __init__(self, logdir, step):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None 
        self._scalars = {} # 记录标量数据
        self._images = {} # 记录图像数据
        self._videos = {} # 记录视频数据
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        # 控制台打印所有标量数据
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            # 将输入的视频从归一化转化为 0-255 的 uint8 格式
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            # B是批次大小，T是帧数，H是高度，W是宽度，C是通道数
            B, T, H, W, C = value.shape
            # 合并B和W维度，并调整维度顺序
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}
    # 计算每秒帧数（FPS）
    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        self._writer.add_scalar("scalars/" + name, value, step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        self._writer.add_video(name, value, step, 16)


def simulate(
    agent,
    envs,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
    is_tuple=False,
    head_infos=None,
):
    # initialize or unpack simulation state
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
        reward = [0] * len(envs)
    else:
        step, episode, done, length, obs, agent_state, reward = state
    # 指定步数或指定回合数就跑够对应的数量
    while (steps and step < steps) or (episodes and episode < episodes):
        # reset envs if necessary
        if done.any(): 
            # 异步环境重置 
            indices = [index for index, d in enumerate(done) if d]
            results = [envs[i].reset() for i in indices]
            results = [r() for r in results]
            # 从重置的环境中更新观测
            for index, result in zip(indices, results):
                t = result.copy()
                # 数据格式转换
                t = {k: convert(v) for k, v in t.items()}
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)
                # replace obs with done by initial state
                obs[index] = result
        # step agents
        # 过滤掉日志信息，按键值堆叠观测
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        # 通过agent来获取动作
        action, agent_state = agent(obs, done, agent_state)
        # 将动作以envs数量的列表形式返回
        if isinstance(action, dict):
            # action = [
            #     {k: np.array(action[k][i].detach().cpu()) for k in action}
            #     for i in range(len(envs))
            # ]
            new_action_list = []
            # 遍历每个环境
            for i in range(len(envs)):
                # 每个环境的动作是一个字典
                env_dict = {}
                # 遍历动作字典的每个键（比如 "move", "attack"）
                for k in action:
                    # 当前环境、当前动作键对应的动作数据
                    act_data = action[k][i]
                    act_data_tuple = action[k]
                    # 判断是否是元组（多头动作）
                    if is_tuple and isinstance(act_data_tuple, tuple):
                        converted = tuple(np.array(ac.squeeze(0).detach().cpu()) for ac in act_data_tuple)
                    else:
                        converted = np.array(act_data.detach().cpu())
                    # 存入当前环境的动作字典
                    env_dict[k] = converted
                # 把当前环境的字典加入列表
                new_action_list.append(env_dict)
            # 最终结果
            action = new_action_list
        # 如果action本身是tuple，则将每个元素转换为numpy数组
        elif isinstance(action, tuple):
            action = [tuple(np.array(a[i].detach().cpu()) for a in action) for i in range(len(envs))]
        else:
            action = np.array(action)
        # 保证动作长度与环境数量一致
        assert len(action) == len(envs)
        # step envs
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum()) # 只有done为True时才增加episode计数
        length += 1
        step += len(envs) # 每次step都会增加envs数量的步数
        length *= 1 - done
        # add to cache
        for a, result, env in zip(action, results, envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            # transition包含了观测值，动作，奖励和折扣
            transition = o.copy()
            # 需要判断动作是否为字典类型
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            # 判断transition["action"]是否是tuple，如果是需要将其拼接
            if isinstance(transition["action"], tuple):
                transition["action"] = np.concatenate([a.flatten() for a in transition["action"]])
            # 添加到经验回放缓冲区
            add_to_cache(cache, env.id, transition)

        if done.any(): # 只有环境结束后才尝试记录日志
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                # 每一次完成后保存
                save_episodes(directory, {envs[i].id: cache[envs[i].id]}, is_tuple=is_tuple, head_infos=head_infos)
                # 计算该回合的长度
                length = len(cache[envs[i].id]["reward"]) - 1
                # 计算该回合的总奖励
                score = float(np.array(cache[envs[i].id]["reward"]).sum())
                if "image" in cache[envs[i].id]:
                    video = cache[envs[i].id]["image"]
                # record logs given from environments
                for key in list(cache[envs[i].id].keys()):
                    if "log_" in key:
                        logger.scalar(
                            key, float(np.array(cache[envs[i].id][key]).sum())
                        )
                        # log items won't be used later
                        cache[envs[i].id].pop(key)

                # 训练模式清理超量经验并记录日志
                if not is_eval:
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar(f"dataset_size", step_in_dataset)
                    logger.scalar(f"train_return", score)
                    logger.scalar(f"train_length", length)
                    logger.scalar(f"train_episodes", len(cache))
                    logger.write(step=logger.step)
                # 评估模式累计奖励并记录日志
                else:
                    if not "eval_lengths" in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_done = False
                    # start counting scores for evaluation
                    eval_scores.append(score)
                    eval_lengths.append(length)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    if "image" in cache[envs[i].id]:
                        logger.video(f"eval_policy", np.array(video)[None])
                    # 当达到指定的评估回合数时记录日志
                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar(f"eval_return", score)
                        logger.scalar(f"eval_length", length)
                        logger.scalar(f"eval_episodes", len(eval_scores))
                        logger.write(step=logger.step)
                        eval_done = True
    # 评估模式下，只保留最后一个元素以节省内存
    if is_eval:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)

# 添加到缓存根据缓存结构cache和经验片段标识id
def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))

# 从缓存中删除超过指定数据集大小的 episodes，保证缓存大小不超过 dataset_size
def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0 #记录当前缓存的总有效步数
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset

# 类型转换函数
def convert(value, precision=32):
    if isinstance(value, tuple):
        #  对tuple中的每个元素进行类型转换
        return tuple(convert(v, precision) for v in value)
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)

# 保存 episodes 到指定目录
def save_episodes(directory, episodes, is_tuple=False, head_infos=None):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    total_action_dim = 0
    if is_tuple and head_infos is not None:
        for head_info in head_infos:
            if "out_dim" not in head_info:
                raise ValueError(f"head_infos 中动作头 {head_info.get('name')} 缺少 out_dim")
            total_action_dim += head_info["out_dim"]
    for filename, episode in episodes.items():
        # if is_tuple:
        #     actions_np = []
        #     for act_tuple in episode["action"]:
        #         if not act_tuple:
        #             flat = np.zeros(total_action_dim, dtype=np.float32)
        #         else:
        #             flat = np.concatenate([a.flatten() for a in act_tuple])
        #         actions_np.append(flat)
        #     episode["action"] = np.array(actions_np)
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True

# 样本生成器
def from_generator(generator, batch_size):
    # 无限循环生成批量样本
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data

# 采样并拼接
def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)
    # 加权采样，长度越长的 episode 被采样的概率越大
    while True:
        size = 0
        ret = None
        # 取任意一个经验类型回去该episode的长度
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        # 权重归一化：每个经验的权重=经验长度/所有经验长度之和
        p = p / np.sum(p)
        while size < length:
            episode = np_random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                # 若片段包含“is_first”字段，则将“is_first”字段设为True，即将片段第一步设为True
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)].copy(), axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret

# 从给定目录加载 episodes（逆序，根据最新时间戳）
# 从给定目录加载 episodes（逆序，根据最新时间戳）
def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes

# def load_episodes(
#     directory, 
#     limit=None, 
#     reverse=True, 
#     head_infos=None
# ):
#     """
#     加载 episodes 并根据 autoregressive_maps 还原多动作头 tuple 结构。
    
#     Args:
#         directory: 保存 npz 文件的目录
#         limit: 最大加载步数（超过则停止）
#         reverse: 是否按文件时间戳逆序加载（优先加载最新）
#         head_infos: 动作头元信息（需包含每个头的 out_dim），格式如：
#             [{"name": "1", "type": "categorical", "out_dim": 1}, ...]
    
#     Returns:
#         episodes: OrderedDict，key 是文件名，value 是还原后的 episode 字典
#     """
#     directory = pathlib.Path(directory).expanduser()
#     episodes = collections.OrderedDict()
#     total = 0  # 累计加载的步数
    
#     # 1. 预处理动作头维度（从 autoregressive_maps 提取每个头的 out_dim，确定切分规则）
#     # 若没有 autoregressive_maps，不还原动作结构（保持原始加载）
#     action_head_dims = []
#     if head_infos is not None:
#         for head_info in head_infos:
#             # 确保每个动作头有 out_dim（离散动作的维度）
#             if "out_dim" not in head_info:
#                 raise ValueError(f"head_infos 中动作头 {head_info.get('name')} 缺少 out_dim")
#             action_head_dims.append(head_info["out_dim"])
#         # 计算 flatten 动作的总长度（用于校验）
#         total_action_dim = sum(action_head_dims)
#         print(f"动作头维度配置：{action_head_dims}，总 flatten 长度：{total_action_dim}")
    
#     # 2. 遍历文件加载 episode
#     file_order = reversed(sorted(directory.glob("*.npz"))) if reverse else sorted(directory.glob("*.npz"))
#     for filename in file_order:
#         try:
#             # 加载 npz 文件（np.load 返回的是 NpzFile 对象，需转成字典）
#             with filename.open("rb") as f:
#                 npz_data = np.load(f)
#                 episode = {k: npz_data[k].copy() for k in npz_data.keys()}  # 复制避免只读问题
#             npz_data.close()  # 关闭文件释放资源
#         except Exception as e:
#             print(f"跳过损坏的 episode 文件 {filename.name}：{str(e)}")
#             continue
        
#         # 3. 若有 autoregressive_maps，还原动作的多头 tuple 结构
#         if head_infos is not None and "action" in episode:
#             flatten_actions = episode["action"]  # 加载的 flatten 动作数组，shape: [steps, total_action_dim]
#             num_steps = len(flatten_actions)     # 该 episode 的总步数
            
#             # 校验 flatten 动作长度是否与配置匹配
#             if flatten_actions.ndim != 2:
#                 raise ValueError(f"动作数组需为 2 维（[steps, total_dim]），当前维度：{flatten_actions.ndim}")
#             if flatten_actions.shape[1] != total_action_dim:
#                 raise ValueError(
#                     f"动作 flatten 长度不匹配：配置总长度 {total_action_dim}，实际 {flatten_actions.shape[1]}"
#                     f"（文件：{filename.name}）"
#                 )
            
#             # 逐步切分动作，还原成 tuple 结构
#             restored_actions = []
#             for step in range(num_steps):
#                 flatten_act = flatten_actions[step]  # 当前步的 flatten 动作：shape [total_action_dim,]
#                 current_pos = 0  # 切分的当前位置
#                 head_acts = []   # 存储当前步的每个动作头
                
#                 # 按动作头维度依次切分
#                 for head_dim in action_head_dims:
#                     # 切分当前动作头的片段（注意：若保存时是 one-hot，这里是 one-hot 数组；若保存时是索引，这里是索引）
#                     head_act = flatten_act[current_pos:current_pos + head_dim]
#                     # 动作头是 one-hot 向量，保持数组形式
#                     head_acts.append(head_act)
                
#                 # 转成 tuple（匹配环境要求的动作类型）
#                 restored_actions.append(tuple(head_acts))
            
#             # 替换 episode 中的动作字段为还原后的 tuple 列表
#             episode["action"] = restored_actions
#             print(f"还原文件 {filename.name} 的动作：{num_steps} 步，每步 {len(action_head_dims)} 个动作头")
        
#         # 4. 记录 episode 并累计步数（停止条件）
#         # 提取文件名（不含后缀）作为 key
#         file_key = str(os.path.splitext(os.path.basename(filename))[0])
#         episodes[file_key] = episode
        
#         # 累计步数（episode["reward"] 长度 = 步数 + 1，故减 1）
#         step_count = len(episode["reward"]) - 1 if "reward" in episode else 0
#         total += step_count
        
#         # 若达到 limit，停止加载
#         if limit is not None and total >= limit:
#             print(f"已加载 {total} 步（达到 limit {limit}），停止加载")
#             break
    
#     print(f"加载完成：共 {len(episodes)} 个 episode，累计 {total} 步")
#     return episodes

# 采样分布类，使用给定的分布进行采样
class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        # 未定义的属性/方法转发给原始分布
        return getattr(self._dist, name)

    def mean(self):
        # 用样本均值近似分布均值
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        # 取概率最大的样本
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)

# 支持梯度回传的One-Hot类别分布
class OneHotDist(torch.distributions.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0, mask=None):
        # 保存原始 logits 和参数
        if logits is not None:
            self._original_logits = logits
        else:
            self._original_logits = None

        self._unimix_ratio = unimix_ratio
        self.mask = mask

        # 初始化时先处理一次 logits
        logits, probs = self._apply_mask_and_unimix(logits, probs, mask)

        super().__init__(logits=logits, probs=probs)

    def _apply_mask_and_unimix(self, logits, probs, mask):
        # 1. 应用 mask
        if logits is not None and mask is not None:
            logits = logits.masked_fill(~mask.bool(), -float('1e9'))

        # 2. 应用 unimix_ratio
        if logits is not None and self._unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            if mask is not None:
                num_valid = mask.sum(dim=-1, keepdim=True)
                uniform = mask.float() / num_valid.clamp_min(1)
            else:
                uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = probs * (1.0 - self._unimix_ratio) + uniform * self._unimix_ratio
            probs = probs.clamp_min(1e-9)  # 防止数值问题
            logits = torch.log(probs)
            probs = None  # 用 logits 初始化父类

        return logits, probs

    def set_mask(self, new_mask):
        """
        用新的 mask 更新分布的 logits/probs，
        基于初始化时保存的原始 logits 和 unimix_ratio
        """
        if self._original_logits is None:
            raise ValueError("OneHotDist 初始化时必须提供 logits 才能使用 set_mask")

        # 基于原始 logits 重新计算
        new_logits, new_probs = self._apply_mask_and_unimix(
            self._original_logits, None, new_mask
        )

        # 更新 mask
        self.mask = new_mask

        # 调用父类的构造函数更新分布
        super().__init__(logits=new_logits, probs=new_probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample

    def log_prob(self, value):
        logp = super().log_prob(value)
        if self.mask is not None:
            # 1. 获取动作索引：最后一维是one-hot，argmax得到前n维（batch维度）的动作索引
            action_index = value.argmax(dim=-1)  # 形状：(d1, d2, ..., dn)
            
            # 2. 生成所有batch维度的索引网格（适配多维度batch）
            # 例如，若action_index形状是(T, B)，则生成t_indices (T,1) 和 b_indices (1,B)
            batch_dims = action_index.shape  # 多维度batch的形状，如(T, B)
            # 为每个batch维度生成索引（如T维度：0~T-1；B维度：0~B-1）
            indices = [torch.arange(dim_size, device=action_index.device) for dim_size in batch_dims]
            # 生成网格索引（meshgrid返回的元组中，每个元素形状与action_index一致）
            grid_indices = torch.meshgrid(indices, indexing='ij')  # 结果是元组，每个元素形状为(d1, d2, ..., dn)
            
            # 3. 用多维度索引获取每个位置的有效性（valid）
            # mask的形状需与value一致：(d1, d2, ..., dn, num_actions)
            # 通过网格索引+action_index索引mask，得到形状为(d1, d2, ..., dn)的valid
            valid = self.mask[grid_indices + (action_index,)]  # 拼接网格索引和动作索引
            
            # 4. 惩罚无效动作的对数概率
            logp = logp.masked_fill(~valid.bool(), -1e4)
        return logp

class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits # 类别分布的logits
        self.probs = torch.softmax(logits, -1)
        # 创建255个均匀分布的桶，覆盖从low到high的范围
        self.buckets = torch.linspace(low, high, steps=255, device=device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        # 计算分布均值（概率加权桶中心，再转换回原来的值）
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)

# 均方误差分布类
class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        # 判断形状是否相同，不相同则打印两者的形状
        assert self._mode.shape == value.shape  , f"{self._mode.shape} != {value.shape}"
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2) - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=True):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = to_np(norm)
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0) if value is not None else None
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            output_val = outputs[j][key]
                            output_val_processed = output_val if output_val is not None else torch.empty(0, dtype=torch.float32)
                            last_val = last[j][key]
                            last_val_processed = last_val.unsqueeze(0) if last_val is not None else torch.empty(0, dtype=torch.float32)
                            outputs[j][key] = torch.cat(
                                [output_val_processed, last_val_processed], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs

# 周期频率触发
class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count

# 单次触发
class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False

# 条件触发
class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=None
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)
