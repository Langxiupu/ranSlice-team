import argparse
import functools
import os
import pathlib
import sys
import gym
from gym.spaces import Tuple

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset, autoregressive_maps=None, head_infos=None):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every) # 日志频率
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio) # 训练频率
        self._should_pretrain = tools.Once() # 预训练一次
        self._should_reset = tools.Every(config.reset_every) # 重置频率
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat)) # 探索直到指定步数
        self._metrics = {}
        self.autoregressive_maps = autoregressive_maps
        self.head_infos = head_infos
        # this is update step
        self._step = logger.step // config.action_repeat # 智能体与环境的总交互步数,考虑帧跳过
        self._update_count = 0
        self._dataset = dataset # 经验回放数据集
        self._wm = models.WorldModel(obs_space, act_space, self._step, config, autoregressive_maps=self.autoregressive_maps, head_infos=self.head_infos) # 世界模型
        self._task_behavior = models.ImagBehavior(config, self._wm, autoregressive_maps=self.autoregressive_maps, head_infos=self.head_infos) # 任务行为模型
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm) # 优化模型的执行效率
            self._task_behavior = torch.compile(self._task_behavior) 
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        # 根据配置文件的expl_behavior选择探索行为
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior, # 贪婪行为
            random=lambda: expl.Random(config, act_space), # 随机行为
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward), # 计划探索行为
        )[config.expl_behavior]().to(self._config.device)

    # 智能体与环境交互
    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            # 根据是否预训练决定训练步数
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                # 从训练集中选取一批数据进行训练
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            # 根据是否到达日志记录间隔来记录日志
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                # 视频日志
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)
            
        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state
    
    def _policy(self, obs, state, training):
        # 初始化状态
        if state is None:
            latent = action = None
        else:
            latent, action = state # 解构state为潜在状态(latent)和上一动作(action)
        obs = self._wm.preprocess(obs) # 预处理观测数据
        embed = self._wm.encoder(obs) # 通过编码器提取低维特征
        # 如果动作类型是tuple，需要尝试将每一个元素拼接起来
        if isinstance(action, tuple):
            action = torch.cat(action, dim=-1)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"]) # 根据潜在状态、动作和观测数据更新潜在状态
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent) # 合并确定性和随机性特征
        if not training: # 评估模式 -> 选择确定性动作
            actor = self._task_behavior.actor(feat)
            action, logprob = self._mode(actor, self.head_infos)
        elif self._should_expl(self._step): # 探索阶段 -> 用探索策略
            actor = self._expl_behavior.actor(feat)
            action, logprob = self._sample(actor, self.head_infos)
        else: # 训练模式 -> 用任务策略
            actor = self._task_behavior.actor(feat) 
            action, logprob = self._sample(actor, self.head_infos)
        # 切断梯度回传
        latent = {k: v.detach() for k, v in latent.items()}
        if not isinstance(action, tuple):
            action = action.detach()
        else:
            action = tuple(a.detach() for a in action)        
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        # 新状态 = (更新后的潜在状态, 当前动作)
        state = (latent, action)
        return policy_output, state
    # 采样函数，主要针对单动作头和多动作头两种情况
    def _sample(self, actor, head_infos=None):     
        """
        根据 actor 类型（单动作头/多动作头）、自回归依赖生成动作和 logprob，
        掩码生成逻辑对齐随机智能体：按依赖计算 available_acts，生成 (N, action_dim) 批量掩码。

        参数:
            actor: 多动作头时为 dict {head_name: distribution, ...}（需支持 set_mask 方法）；
                单动作头时为 distribution 对象；
                所有分布需具备 logits 属性（获取批量大小 N）和 set_mask 方法（设置掩码）。

        返回:
            action: 拼接后的动作张量（多动作头）或单动作张量，形状 (N, total_out_dim) 或 (N, act_dim)
            logprob: 累计 log 概率，形状 (N,)
        """
        # 1. 单动作头处理（无自回归，直接采样）
        if not isinstance(actor, dict):
            action = actor.sample()
            logprob = actor.log_prob(action)
            return action, logprob

        # 2. 多动作头自回归处理
        # 2.1 预处理：动作头名称与索引映射（确保与 autoregressive_maps 顺序一致）
        head_names = list(actor.keys())  # 动作头名称列表
        num_heads = len(head_names)
        # 校验：自回归配置与动作头数量一致，且分布支持 set_mask
        assert hasattr(self, "autoregressive_maps") and len(self.autoregressive_maps) == num_heads, \
            f"autoregressive_maps 长度({len(self.autoregressive_maps)})需与动作头数量({num_heads})一致"

        # 2.2 初始化：批量大小 N、动作列表、累计 logprob
        first_dist = actor[head_names[0]]
        N = first_dist.logits.shape[0]  # 批量大小（从第一个动作头的 logits 提取）
        sampled_actions = []  # 存储每个动作头的采样结果（按索引顺序）
        total_logprob = torch.zeros(N, device=first_dist.logits.device)  # 累计 logprob

        # 2.3 遍历每个动作头，按自回归依赖生成掩码并采样
        for head_idx in range(num_heads):
            # 当前动作头基础信息
            current_head_name = head_names[head_idx]
            current_dist = actor[current_head_name]
            current_autoregressive = self.autoregressive_maps[head_idx]  # 依赖配置（如 [-1]、[1]）
            # 获取当前动作头的动作维度（action_dim）：优先从分布 logits 提取，兼容 gym 空间
            action_dim = current_dist.logits.shape[-1] if head_infos[head_idx]["type"] == "categorical" else None # logits 形状 (N, action_dim)
            # 2.4 计算当前动作头的 available_acts
            # 初始可用动作数：每个环境都为 full 动作维度（N 个环境，每个环境初始可用 action_dim 个动作）
            available_acts = torch.full((N,), action_dim, device=current_dist.logits.device) if head_infos[head_idx]["type"] == "categorical" else None
            if current_autoregressive and current_autoregressive[0] != -1:
                # 遍历当前动作头的所有依赖（支持多依赖，如 [1,2] 依赖索引1和2的动作头）
                for dep_idx in current_autoregressive:
                    # 依赖索引必须小于当前索引（自回归需前序动作已采样）
                    assert dep_idx < head_idx, \
                        f"动作头{current_head_name}（索引{head_idx}）依赖的索引{dep_idx}需小于当前索引"
                    # 获取依赖动作头的已采样动作（shape: (N, dep_action_dim)）
                    dep_act = sampled_actions[dep_idx]
                    # 提取每个环境的动作类别（one-hot 动作取 argmax，得到 (N,)）
                    dep_act_val = dep_act.argmax(dim=-1)  # 每个环境的依赖动作值（如 0、1、2）
                    # 调整可用动作数：available_acts = 初始值 - 依赖动作值（对齐随机智能体逻辑）
                    available_acts = available_acts - dep_act_val
                    # 确保可用动作数不为负（避免掩码生成错误）
                    available_acts = torch.clamp(available_acts, min=1)  # 至少保留1个可用动作
            # 2.5 生成批量掩码（shape: (N, action_dim)）
            # 生成动作索引（0 ~ action_dim-1，shape: (action_dim,)）
            act_indices = torch.arange(action_dim, device=current_dist.logits.device) if head_infos[head_idx]["type"] == "categorical" else None
            # 批量掩码：每个环境的前 available_acts[n] 个动作为 1（可用），其余为 0（禁用）
            # available_acts.unsqueeze(1) 扩展为 (N,1)，与 act_indices 广播对比
            mask = (act_indices < available_acts.unsqueeze(1)).float()  if head_infos[head_idx]["type"] == "categorical" else None # (N, action_dim)

            # 2.6 应用掩码并采样
            current_dist.set_mask(mask) if mask is not None else None # 给当前分布设置批量掩码
            current_act = current_dist.sample()  # 采样当前动作（shape: (N, action_dim)）
            current_logprob = current_dist.log_prob(current_act)  # 计算当前动作 logprob（shape: (N,)）

            # 存储结果
            sampled_actions.append(current_act)
            total_logprob += current_logprob  # 累计 logprob（联合概率=各动作概率乘积→log求和）

        # 3. 拼接多动作头结果
        logprob = total_logprob  # (N,) → 累计联合 logprob

        return tuple(sampled_actions), logprob
    
    def _mode(self, actor, head_infos=None):     
        """
        根据 actor 类型（单动作头/多动作头）、自回归依赖获取动作分布的 mode（最可能动作），
        掩码生成逻辑与 _sample 完全对齐：按依赖计算 available_acts，生成 (N, action_dim) 批量掩码。

        参数:
            actor: 多动作头时为 dict {head_name: distribution, ...}（需支持 set_mask 方法）；
                单动作头时为 distribution 对象；
                所有分布需具备 logits 属性（获取批量大小 N）和 set_mask 方法（设置掩码）。

        返回:
            action: 拼接后的动作张量（多动作头）或单动作张量，形状 (N, total_out_dim) 或 (N, act_dim)
            logprob: 累计 log 概率（mode 动作的概率），形状 (N,)（可选，评估模式常用）
        """
        # 1. 单动作头处理（无自回归，直接取 mode）
        if not isinstance(actor, dict):
            # 取分布的 mode（最可能动作），需确保分布实现 mode() 方法
            assert hasattr(actor, "mode"), f"单动作头分布需实现 mode() 方法"
            action = actor.mode()
            # 计算 mode 动作的 logprob（评估时需用到）
            logprob = actor.log_prob(action)
            return action, logprob
        # 2. 多动作头自回归处理
        # 2.1 预处理：动作头名称与索引映射（确保与 autoregressive_maps 顺序一致）
        head_names = list(actor.keys())  # 动作头名称列表（如 ["1", "2", "3", "4"]）
        num_heads = len(head_names)
        # 校验：自回归配置、分布方法兼容性
        assert hasattr(self, "autoregressive_maps") and len(self.autoregressive_maps) == num_heads, \
            f"autoregressive_maps 长度({len(self.autoregressive_maps)})需与动作头数量({num_heads})一致"

        # 2.2 初始化：批量大小 N、动作列表、累计 logprob
        first_dist = actor[head_names[0]]
        N = first_dist.logits.shape[0]  # 批量大小（从第一个动作头的 logits 提取）
        mode_actions = []  # 存储每个动作头的 mode 动作（按索引顺序）
        total_logprob = torch.zeros(N, device=first_dist.logits.device)  # 累计 mode 动作的 logprob

        # 2.3 遍历每个动作头，按自回归依赖生成掩码并取 mode
        for head_idx in range(num_heads):
            # 当前动作头基础信息
            current_head_name = head_names[head_idx]
            current_dist = actor[current_head_name]
            current_autoregressive = self.autoregressive_maps[head_idx]  # 依赖配置（如 [-1]、[1]）
            action_dim = current_dist.logits.shape[-1] if head_infos[head_idx]["type"] == "categorical" else None  # 当前动作头的动作维度（N, action_dim）

            # 2.4 计算 available_acts（与 _sample 完全一致）
            # 初始可用动作数：每个环境为 full 动作维度
            available_acts = torch.full((N,), action_dim, device=current_dist.logits.device) if head_infos[head_idx]["type"] == "categorical" else None

            # 处理自回归依赖：根据前序动作调整可用动作数
            if current_autoregressive and current_autoregressive[0] != -1:
                for dep_idx in current_autoregressive:
                    assert dep_idx < head_idx, \
                        f"动作头{current_head_name}（索引{head_idx}）依赖的索引{dep_idx}需小于当前索引"
                    # 获取前序动作头的 mode 动作
                    dep_act = mode_actions[dep_idx]
                    # 提取依赖动作的类别（one-hot 取 argmax）
                    dep_act_val = dep_act.argmax(dim=-1)  # (N,)
                    # 调整可用动作数（对齐随机智能体逻辑）
                    available_acts = available_acts - dep_act_val
                    available_acts = torch.clamp(available_acts, min=1)  # 至少保留1个可用动作

            # 2.5 生成批量掩码（与 _sample 完全一致）
            act_indices = torch.arange(action_dim, device=current_dist.logits.device) if head_infos[head_idx]["type"] == "categorical" else None
            mask = (act_indices < available_acts.unsqueeze(1)).float()  if head_infos[head_idx]["type"] == "categorical" else None # (N, action_dim)

            # 2.6 应用掩码并取 mode（核心差异：sample → mode）
            current_dist.set_mask(mask)  if mask is not None else None # 应用批量掩码
            current_mode = current_dist.mode()  # 取最可能动作（而非随机采样）
            current_logprob = current_dist.log_prob(current_mode)  # 计算 mode 动作的 logprob

            # 存储结果
            mode_actions.append(current_mode)
            total_logprob += current_logprob  # 累计 mode 动作的联合 logprob

        # 3. 拼接多动作头结果

        logprob = total_logprob  # (N,) → 累计联合 logprob

        return tuple(mode_actions), logprob


    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets) # 更新模型训练指标
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1]) # 在想象轨迹上训练任务策略
        if self._config.expl_behavior != "greedy": # 如果不是贪婪行为
            # 训练探索行为模型
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
    
    def predict_next_obs(self, obs, action, is_first, is_terminal):
        return self._task_behavior.predict_next_obs(obs, action, is_first, is_terminal)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

# 从经验回放缓冲区生成训练数据集
def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "ale":
        from envs.ale import ALEEnv
        env = ALEEnv(
            action_repeat=config.action_repeat,
            frame_size=config.size,
            gray_scale=config.grayscale,
            noop_max=config.noops,
            life_done=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            obs_type=config.obs_type,
            seed=0,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "test":
        from envs.test import DummyMultiHeadEnv
        env = DummyMultiHeadEnv()
        env = wrappers.MultiHeadAction(env)
    elif suite == "ranSlice":
        from envs.env import LEORANSlicing
        env = LEORANSlicing("env_config.json", training=config.training)
        env = wrappers.MultiHeadAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    # 将配置参数转化为环境步数
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir) # 统计已有的经验回放数据集的步数
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    # 根据配置的envs数量创建训练和评估环境
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    eval_policy_env = make_env(config, "eval_policy", 0)
    # 如果配置了并行处理，则使用Parallel包装器
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    head_infos = None
    autoregressive_map = None
    if config.multi_head:
        head_infos = train_envs[0].action_head_infos
        print("Head Info", head_infos)
        autoregressive_map = train_envs[0].action_autoregressive_map
        print("Autoregressive Map", autoregressive_map)
    print("Action Space", acts)
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    is_tuple = False
    if isinstance(acts, Tuple) and config.multi_head:
        is_tuple = True
        # 遍历 tuple 中的每个动作空间，将他们的维度相加得到总维度
        config.num_actions = sum(acts[i].n if hasattr(acts[i], "n") else acts[i].shape[0] for i in range(len(acts)))
    else:
        # 根据动作空间是离散还是连续，设置动作数量或维度
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    print("Num Actions", config.num_actions)

    state = None

    if config.eval_policy:
        # 如果我们是处于评估策略模式，我们直接加载原有的模型
        print("Evaluate policy.")
        # 不需要加载训练数据集，直接按已有环境评估一个episode
        agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        dataset=None,
        autoregressive_maps=autoregressive_map if autoregressive_map else None,
        head_infos=head_infos,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)
        if (logdir / "latest.pt").exists():
            checkpoint = torch.load(logdir / "latest.pt")
            agent.load_state_dict(checkpoint["agent_state_dict"])
            tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
            agent._should_pretrain._once = False
        obs = eval_policy_env.reset()
        reward_total = 0
        done = False
        is_first = True
        state = None
        while not done:
            # 得到的obs中的每个元素都需要加上batch维度
            obs_batch = {k: np.expand_dims(v, axis=0) for k, v in obs.items()}
            with torch.no_grad():
                policy_output, state = agent(
                    obs_batch,
                    np.array([done]),
                    state,
                    training=False,
                )
                # 需要将动作从tensor转化为numpy格式，并去掉batch维度
                # 首先选取policy_output['action'],这是一个tuple，里面每个元素都是tensor，第一个维度是batch维度
                if isinstance(policy_output["action"], tuple):
                    action = tuple(
                        act.cpu().numpy()[0] for act in policy_output["action"]
                    )
                else:
                    action = policy_output["action"].cpu().numpy()[0]
                # 我们重新设置一个新的动作变量，将所有动作展开成一维并拼接
                new_action = None
                if isinstance(action, tuple):
                    new_action = np.concatenate(
                        [act.reshape(-1) for act in action], axis=0
                    )
                # 预测下一步的obs
                next_obs = agent.predict_next_obs(obs, new_action, is_first, done)
                is_first = False
                eval_policy_env.set_next_obs(next_obs)
                input_action ={"action": action}
                obs, reward, done, info = eval_policy_env.step(input_action)
                reward_total += reward
        print(f"Total reward: {reward_total}")
        return

    if not config.offline_traindir:
        # 正式训练前预填充数据集,采用随机策略。
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if isinstance(acts, Tuple): 
            # random_actor是acts里每个动作空间的独立分布的组合,目前支持的是One-Hot分布
            random_actor = []
            for i in range(len(acts)):
                if hasattr(acts[i], "n"):
                    random_actor.append(tools.OneHotDist(torch.zeros(acts[i].n).repeat(config.envs, 1)))
                else:
                    random_actor.append(torchd.independent.Independent(
                        torchd.uniform.Uniform(
                            torch.tensor(acts[i].low).repeat(config.envs, 1),
                            torch.tensor(acts[i].high).repeat(config.envs, 1),
                        ),
                        1,
                    ))
        elif hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )
        
        def random_agent(o, d, s):
            # 如果random_actor是tuple，说明动作空间是多个独立的动作空间
            if is_tuple:
                action = []
                N = random_actor[0].logits.shape[0]
                logprob = torch.zeros(N, device=random_actor[0].logits.device)
                for i in range(len(acts)):
                    # 1. 初始可用动作数：带批量维度，每个环境初始都是 full 动作数量
                    action_dim = acts[i].n if hasattr(acts[i], "n") else acts[i].shape[0]
                    available_acts = torch.full((N,), action_dim, device=random_actor[i].logits.device) if hasattr(acts[i], "n") else None
                    # 判断依赖关系中的索引,遍历autoregressive_map
                    if autoregressive_map and autoregressive_map[i] is not None:
                        for index in autoregressive_map[i]:
                            # 判断是否为 -1，如果是，则说明该动作不依赖其他动作
                            if index == -1:
                                break
                            # 如果依赖的动作是tuple中的某个动作，则获取该动作的值
                            else:
                                ac = action[index] # [N, action_dim]
                                # 提取每个环境的动作值
                                ac_val = ac.argmax(dim=-1) # [N,]
                                available_acts = available_acts - ac_val
                    # 生成批量mask：(N, action_dim)
                    act_indices = torch.arange(action_dim, device=available_acts.device) if hasattr(acts[i], "n") else None
                    # mask: (N, action_dim)，每个环境的前 available_acts[n] 个动作为 1，其余为 0
                    mask = (act_indices < available_acts.unsqueeze(1)).float() if hasattr(acts[i], "n") else None
                    random_actor[i].set_mask(mask) if mask is not None else None
                    # 采样动作
                    ac = random_actor[i].sample()
                    # 计算logprob
                    logp = random_actor[i].log_prob(ac)
                    # 将动作添加到动作列表中
                    action.append(ac)
                    # 计算logprob之和
                    logprob += logp
                return {"action": tuple(action), "logprob": logprob}, None
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
            is_tuple=is_tuple,
            head_infos=head_infos,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
        autoregressive_maps=autoregressive_map if autoregressive_map else None,
        head_infos=head_infos,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
                is_tuple=is_tuple,
                head_infos=head_infos,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
            is_tuple=is_tuple,
            head_infos=head_infos,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    # 递归字典合并函数
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    # 生成具体的参数解析
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
