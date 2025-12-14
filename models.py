import copy
import torch
from torch import nn

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()

# 将奖励线性映射到一个相对稳定的尺度,后续用作归一化。
class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config, autoregressive_maps=None, head_infos=None):
        super(WorldModel, self).__init__()
        self._step = step # 训练步数
        self._use_amp = True if config.precision == 16 else False # 是否使用混合精度
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()} # 提取观测空间各个键值对的形状
        self.shapes = shapes
        self.encoder = networks.MultiEncoder(shapes, **config.encoder) # 多模态编码器，将不同观测（如图像、向量）映射到统一隐空间
        self.embed_size = self.encoder.outdim # 编码器输出的隐空间维度
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        ) # 动态模型，RSSM（Recurrent State Space Model），用于建模环境的动态变化
        self.heads = nn.ModuleDict() # 任务头，包含解码器、奖励预测器、连续性预测器等
        # 根据是否离散状态来计算特征大小
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        # 初始化解码器、奖励预测器、连续性预测器等任务头
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        # 自定义优化器
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data) # 编码观测数据，得到统一的隐空间表示
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                ) # 观察模型，更新后验状态和先验状态
                kl_free = self._config.kl_free # KL散度阈值
                dyn_scale = self._config.dyn_scale # 动态模型的缩放系数
                rep_scale = self._config.rep_scale # 表示学习的缩放系数
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post) # 合并随机和确定性状态
                    feat = feat if grad_head else feat.detach() # 如果不是梯度头，则不计算梯度
                    pred = head(feat) # 通过任务头预测结果
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                # 计算每个任务头的损失
                losses = {}
                for name, pred in preds.items():
                    target = data[name]
                    # 需要self.shapes存在name这个key值并且shape长度等于于2
                    if name in self.shapes and len(self.shapes[name]) == 2:
                        # 保留data[name]的前n-2维度，将最后两维展平
                        target = data[name].reshape(
                            list(data[name].shape[:-2]) + [data[name].shape[-2] * data[name].shape[-1]]
                        )
                    loss = -pred.log_prob(target)
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())
        # 各种损失的统计信息
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        } # 将观测值转换为张量，并移动到指定设备上
        # 判断动作类型是否是tuple，如果是，需要尝试拼接
        if "action" in obs:
            a = obs["action"]
            if isinstance(a, tuple):
                obs["action"] = torch.cat([x for x in a], dim=-1)
            elif a.dim() == 3:
                pass
            else:
                pass
        if "image" in obs:
            obs["image"] = obs["image"] / 255.0 # 将图像观测值归一化到[0, 1]范围
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        # 编码观测并生成初始状态
        data = self.preprocess(data)
        embed = self.encoder(data)

        # 观察模型，获取前6个时间步的状态和动作
        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        # 使用前5个时间步的动作来预测后续的状态
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0
        # 将重建的图像、真实图像和误差拼接在一起
        return torch.cat([truth, model, error], 2)

# 这是一个行为模型，使用想象的状态来生成动作和价值预测
class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, autoregressive_maps=None, head_infos=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model # 依赖的世界模型
        self._autoregressive_maps = autoregressive_maps # 自回归映射
        self._head_infos = head_infos # 头信息
        self._is_multi_head = True if head_infos is not None else False # 是否是多头
            
        # 根据离散特征确定特征大小
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        print("ImagBehavior feat_size:", feat_size) 
        # Actor网络
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
            is_multi_head=self._is_multi_head,
            head_info=self._head_infos,
            autoregressive_maps=self._autoregressive_maps
        )
        # Critic网络（价值网络）
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        # 慢更新目标网络
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp) # 优化器的额外参数
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # 注册一个缓冲区张量 ema_vals，用于保存EMA的统计量（偏移和尺度）
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def predict_next_obs(self, obs: dict, action=None, is_first=True, is_terminal=False):
        """
        给定单步环境观测 obs（字典），使用 world model + imaginated state 预测下一时刻的观测。

        输入:
          - obs: 原始环境观测字典，键和值与 env.observation 格式一致（不要求 batch/time 维度）
          - action: 可选的下一步动作（numpy array 或可广播到动作维度），若为 None 则使用 0 动作
          - is_first: bool，指示该观测是否为序列起始（用于初始化 RSSM）；默认 True
          - is_terminal: bool，指示该观测是否为终止标志；默认 False

        返回:
          - pred_obs: 字典，包含与 obs 中同名键对应的预测下一时刻观测（numpy 数组，shape 与 env observation 一致）

        说明:
          - 该方法在内部会把输入扩展为 (batch=1, time=1, ...)，调用 world model 的 encoder/dynamics/decoder
            完成一步想象（horizon=1），然后返回 decoder 的 mode() 作为下一时刻的观测预测。
          - 不会修改模型自身的训练状态（不会更新参数）。
        """

        import numpy as _np

        wm = self._world_model
        shapes = getattr(wm, "shapes", None)
        if shapes is None:
            raise RuntimeError("world_model.shapes 未定义，无法推断观测键和形状")

        # helper: expand an array-like to (batch=1, time=1, *shape)
        def _expand(x, target_shape):
            arr = _np.array(x)
            # target_shape is a tuple describing the raw obs shape (no batch/time)
            nd = arr.ndim
            if nd == len(target_shape):
                return arr[None, None, ...]
            if nd == len(target_shape) + 1:
                # assume (time, ...)
                return arr[None, ...]
            if nd == len(target_shape) + 2:
                # assume (batch, time, ...)
                return arr
            # allow scalar targets
            if len(target_shape) == 0 and nd == 0:
                return arr.reshape((1, 1))
            raise ValueError(f"输入观测 {arr.shape} 无法扩展到目标形状 {target_shape}")

        data = {}
        # fill data for all keys expected by world model (missing keys -> zeros)
        for k, shape in wm.shapes.items():
            if k in obs:
                try:
                    data[k] = _expand(obs[k], shape)
                except Exception:
                    # 若无法按预期扩展，尝试直接使用原值并 hope for the best
                    data[k] = _np.array(obs[k])[None, None]
            else:
                # create zeros for missing keys
                z = _np.zeros((1, 1) + tuple(shape), dtype=_np.float32)
                data[k] = z

        # ensure action exists
        num_actions = getattr(wm._config, "num_actions", None)
        if "action" not in data:
            if action is None:
                if num_actions is None:
                    # fallback to scalar zero
                    data["action"] = _np.zeros((1, 1), dtype=_np.float32)
                else:
                    data["action"] = _np.zeros((1, 1, num_actions), dtype=_np.float32)
            else:
                a = _np.array(action)
                if a.ndim == 1 and num_actions is not None and a.shape[0] == num_actions:
                    data["action"] = a[None, None, ...]
                elif a.ndim == 0:
                    data["action"] = a[None, None]
                else:
                    # try to reshape/broadcast
                    data["action"] = a.reshape((1, 1) + a.shape)

        # is_first / is_terminal
        data.setdefault("is_first", _np.array(is_first, dtype=_np.bool_).reshape((1, 1)))
        data.setdefault("is_terminal", _np.array(is_terminal, dtype=_np.bool_).reshape((1, 1)))

        # preprocess -> tensors on device
        proc = wm.preprocess(data)

        # encode and obtain posterior for the single step
        with torch.no_grad():
            embed = wm.encoder(proc)  # expects batch,time,...
            post, prior = wm.dynamics.observe(embed, proc["action"], proc["is_first"])  # dicts with time dim

            # take last posterior state (time dim -1)
            init = {k: v[:, -1] for k, v in post.items()}

            # prepare one-step action (use provided action or zero)
            if action is None:
                if num_actions is None:
                    a_next = torch.zeros_like(proc["action"])
                else:
                    a_next = torch.zeros((proc["action"].shape[0], 1, num_actions), device=proc["action"].device)
            else:
                a_np = _np.array(action)
                a_next = torch.tensor(a_np, device=proc["action"].device, dtype=torch.float32)
                if a_next.ndim == 1 and num_actions is not None and a_next.shape[0] == num_actions:
                    a_next = a_next[None, None, ...]
                elif a_next.ndim == 2 and a_next.shape[0] == 1:
                    a_next = a_next[None, ...]
                elif a_next.ndim == 0:
                    a_next = a_next.reshape((1, 1))

            prior_next = wm.dynamics.imagine_with_action(a_next, init)
            feat = wm.dynamics.get_feat(prior_next)  # (batch, time=1, feat)
            obs_dist = wm.heads["decoder"](feat)

            pred = {}
            for k, dist in obs_dist.items():
                # dist.mode() shape: (batch, time, *shape)
                m = dist.mode()
                # convert to numpy and remove batch/time dims
                arr = m.detach().cpu().numpy()
                arr = arr.reshape(arr.shape[2:]) if arr.ndim >= 3 else arr.squeeze()
                pred[k] = arr
        # 这里我们需要将pred中的数据还原为原来的形状，当前是全部展开为一维
        # 需要遍历shapes来还原
        for k, shape in wm.shapes.items():
            if k in pred:
                pred[k] = pred[k].reshape(shape)
        return pred

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target() # 更新慢目标网络
        metrics = {}
        # Actor训练阶段
        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action, autogressive_masks  = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                obs_dist = self._world_model.heads["decoder"](imag_feat)
                restored_obs = {}
                for obs_key, dist in obs_dist.items():  # obs_key: 观测的键（如 "image"、"vec"）
                    restored_obs[obs_key] = dist.mode()  # 取最可能值（推荐）
                reward = objective(imag_feat, imag_state, imag_action) # 计算想象的奖励,利用世界模型的奖励预测头
                actor_dist = self.actor(imag_feat) # 计算actor的动作分布
                if self._is_multi_head:
                    # 多动作头时：各个动作头的熵求和
                    actor_ent = torch.stack([dist.entropy() for dist in actor_dist.values()], dim=-1).sum(-1)
                else:
                    actor_ent = actor_dist.entropy() # 计算actor的熵
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy() # 计算状态的熵
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                    autoregressive_masks=autogressive_masks
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat
        # Value训练阶段
        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            if self._is_multi_head:
                # 多动作头时，记录每个动作头的动作分布
                split_acts = self._split_action(imag_action)
                for i, (head_name, act) in enumerate(split_acts.items()):
                    act_cls = torch.argmax(act, dim=-1).float()
                    metrics[f"imag_action_{head_name}"] = to_np(act_cls)
            else:
                metrics.update(
                    tools.tensorstats(
                        torch.argmax(imag_action, dim=-1).float(), "imag_action"
                    )
                )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:])) # 展开除batch和时间维度外的其他维度
        start = {k: flatten(v) for k, v in start.items()} # 展平初始状态

        def step(prev, _):
            state, _, _, _ = prev
            feat = dynamics.get_feat(state) # 获取特征表示
            inp = feat.detach() # 获取特征表示，detach()用于防止梯度传播
            # 采样动作+获取自回归掩码
            if self._is_multi_head:
                action, step_ar_masks = self._sample(policy(inp),self._head_infos)
            else:
                action, _ = self._sample(policy(inp), None)
                step_ar_masks = torch.zeros_like(action)[:, :1]
            succ = dynamics.img_step(state, action) # 使用动态模型预测下一个状态
            return succ, feat, action, step_ar_masks
        ar_masks_list = None
        # 根据想象轨迹的长度进行静态扫描
        if self._is_multi_head:
            succ, feats, actions, ar_masks_list = tools.static_scan(
                step, [torch.arange(horizon)], (start, None, None, None)
            )
        else:
            succ, feats, actions, _ = tools.static_scan(
                step, [torch.arange(horizon)], (start, None, None, None)
            )
        # 拼接初始状态和预测状态（去掉最后一步）
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        autoregressive_masks = None
        if self._is_multi_head and ar_masks_list is not None:
            # 按动作头名称整理掩码（每个动作头的掩码形状：(horizon, batch, action_dim)）
            head_names = [h["name"] for h in self._head_infos]
            autoregressive_masks = {}
            for head_idx, head_name in enumerate(head_names):
                # 提取每个时间步该动作头的掩码，堆叠为(horizon, batch, action_dim)
                head_masks = torch.stack([step_mask for step_mask in ar_masks_list[head_name]], dim=0)  if self._head_infos[head_idx]["type"] == "categorical" else None
                autoregressive_masks[head_name] = head_masks
        return feats, states, actions, autoregressive_masks

    def _compute_target(self, imag_feat, imag_state, reward):
        # 计算折扣因子（若存在终止预测头"cont"，则动态调整）
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        # 计算 λ-return
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        # 计算权重（折扣因子的累积乘积）
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
        autoregressive_masks=None, #针对多动作头
    ):
        metrics = {}
        inp = imag_feat.detach()  # 固定：actor梯度不回传世界模型，仅优化策略
        policy = self.actor(inp)  # 输出：多动作头→dict{head_name: 分布}, 单动作头→分布对象
        target_stacked = torch.stack(target, dim=1)  # 堆叠Q目标值：(horizon-1, batch, num_q)

        # 步骤1：目标值标准化（可选，基于EMA）
        # 作用：避免奖励尺度差异导致训练不稳定
        if self._config.reward_EMA:
            # 计算EMA统计量（偏移offset+缩放scale），过滤极端值影响
            offset, scale = self.reward_ema(target_stacked, self.ema_vals)
            # 标准化目标值和基准值（V函数输出）
            normed_target = (target_stacked - offset) / scale
            normed_base = (base - offset) / scale
            # 计算优势函数：A = Q_norm - V_norm（减少方差，精准引导策略）
            adv = normed_target - normed_base
            # 记录标准化相关指标（调试用）
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])  # EMA 5%分位数
            metrics["EMA_095"] = to_np(self.ema_vals[1])  # EMA 95%分位数
        else:
            # 未开启EMA：直接用原始值计算优势
            adv = target_stacked - base if "adv" not in locals() else adv

        # 步骤2：计算策略目标（policy_log_prob）
        # 核心：区分单/多动作头，多动作头需拆分动作+聚合log_prob+掩码过滤
        if self._is_multi_head:
            # 2.1 拆分拼接动作：与_sample的拼接顺序严格一致
            split_imag_actions = self._split_action(imag_action)  # dict{head_name: act_tensor}
            total_log_prob = None  # 初始化联合log_prob（各动作头log_prob之和）
            # 遍历每个动作头，计算带掩码的log_prob
            count_idx = 0
            for head_name in split_imag_actions.keys():
                # 获取当前动作头的核心数据
                head_dist = policy[head_name]  # 动作头的分布对象
                head_act = split_imag_actions[head_name]  # 拆分后的动作：(horizon, batch, action_dim)
                head_mask = autoregressive_masks.get(head_name) if self._head_infos[count_idx]["type"] == "categorical" else None  # 自回归掩码
                head_dist.set_mask(head_mask) if head_mask is not None else None  # 应用掩码

                # 2.2 计算原始log_prob（未过滤无效动作）
                # 形状：(horizon, batch) → 每个时间步-批量的动作概率对数
                raw_log_prob = head_dist.log_prob(head_act)
                masked_log_prob = raw_log_prob
                # 2.3 应用自回归掩码：过滤无效动作的log_prob（核心）
                if head_mask is not None:
                    # 验证掩码维度：必须与动作匹配 (horizon, batch, action_dim)
                    assert head_mask.shape == head_act.shape, \
                        f"动作头{head_name}的掩码形状{head_mask.shape}需与动作形状{head_act.shape}一致"
                    masked_log_prob = raw_log_prob
                    metrics[f"head_{head_name}_log_prob"] = to_np(masked_log_prob.mean())  # 记录原始log_prob均值

                # 2.4 聚合联合log_prob：多动作头log_prob求和
                if total_log_prob is None:
                    total_log_prob = masked_log_prob
                else:
                    total_log_prob += masked_log_prob
                count_idx += 1

            # 2.5 关键：截取前horizon-1步（最后一步无后续奖励，不参与更新）
            # 扩展维度：从(horizon-1, batch) → (horizon-1, batch, 1)，匹配target_stacked维度
            policy_log_prob = total_log_prob[:-1].unsqueeze(-1)

        else:
            # 单动作头逻辑（保留原始行为，兼容旧配置）
            # 计算log_prob → 截取前horizon-1步 → 扩展维度
            raw_log_prob = policy.log_prob(imag_action)  # (horizon, batch)
            policy_log_prob = raw_log_prob[:-1].unsqueeze(-1)  # (horizon-1, batch, 1)
            metrics["single_head_log_prob"] = to_np(raw_log_prob.mean())  # 记录单动作头log_prob
        # 步骤3：根据imag_gradient配置计算actor_target（策略优化目标）
        # 三种模式：dynamics（直接用优势）、reinforce（log_prob×优势）、both（混合）
        if self._config.imag_gradient == "dynamics":
            # 模式1：直接最大化优势（适用于确定性策略或简化场景）
            # 注意：优势的时间步已为horizon-1，无需再截取
            actor_target = adv

        elif self._config.imag_gradient == "reinforce":
            # 模式2：经典REINFORCE逻辑（log_prob×优势，最大化高优势动作的概率）
            # 计算优势项：target_stacked（Q值） - V值（基准），detach避免V网络梯度回传
            advantage_term = (target_stacked - self.value(imag_feat[:-1]).mode()).detach()
            actor_target = policy_log_prob * advantage_term  # (horizon-1, batch, num_q)

        elif self._config.imag_gradient == "both":
            # 模式3：混合模式（平衡dynamics的稳定性和reinforce的精准性）
            advantage_term = (target_stacked - self.value(imag_feat[:-1]).mode()).detach()
            reinforce_target = policy_log_prob * advantage_term
            # 混合系数：mix×Q值 + (1-mix)×REINFORCE目标
            mix = self._config.imag_gradient_mix
            actor_target = mix * target_stacked + (1 - mix) * reinforce_target
            metrics["imag_gradient_mix"] = mix  # 记录混合系数

        else:
            raise NotImplementedError(f"不支持的imag_gradient配置：{self._config.imag_gradient}")

        # 步骤4：计算最终actor损失（转为最小化目标+权重过滤）
        # 截取权重前horizon-1步，并扩展维度（与actor_target对齐）
        weights_truncated = weights[:-1]  # (horizon-1, batch, 1)
        # 损失计算：-权重×目标（负号：最大化目标→最小化损失；权重：过滤无效轨迹）
        actor_loss = -weights_truncated * actor_target

        # 步骤5：记录核心损失指标（调试用）
        metrics["actor_loss_raw"] = to_np(actor_loss.mean())  # 原始损失均值
        metrics["policy_log_prob_mean"] = to_np(policy_log_prob.mean())  # 策略log_prob均值
        if "adv" in locals():
            metrics["advantage_mean"] = to_np(adv.mean())  # 优势函数均值

        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            # 每隔一定步数更新一次慢目标网络
            if self._updates % self._config.critic["slow_target_update"] == 0:
                # 使用慢目标网络的更新比例来更新慢目标网络的参数
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
    
    # 采样函数，主要针对单动作头和多动作头两种情况
    def _sample(self, actor, head_infos=None):     
        """
        根据 actor 类型（单动作头/多动作头）、自回归依赖生成动作，并返回掩码。

        参数:
            actor: 多动作头时为 dict {head_name: distribution, ...}（需支持 set_mask 方法）；
                单动作头时为 distribution 对象；

        返回:
            action: 拼接后的动作张量（多动作头）或单动作张量，形状 (N, total_out_dim) 或 (N, act_dim)
            autoregressive_masks (dict, optional): 每个动作头的掩码，形状 (N, action_dim)
        """
        # 单动作头处理
        if not isinstance(actor, dict):
            action = actor.sample()
            return action, None  # 单动作头不返回掩码

        # 多动作头自回归处理
        head_names = list(actor.keys())
        num_heads = len(head_names)
        assert hasattr(self, "_autoregressive_maps") and len(self._autoregressive_maps) == num_heads, \
            f"autoregressive_maps 长度({len(self._autoregressive_maps)})需与动作头数量({num_heads})一致"

        first_dist = actor[head_names[0]]
        N = first_dist.logits.shape[0]  # 批量大小
        sampled_actions = []
        autoregressive_masks = {}  # 保存每个动作头的掩码

        for head_idx in range(num_heads):
            current_head_name = head_names[head_idx]
            current_dist = actor[current_head_name]
            current_autoregressive = self._autoregressive_maps[head_idx]
            action_dim = current_dist.logits.shape[-1] if head_infos[head_idx]["type"] == "categorical" else None

            # 初始可用动作数
            available_acts = torch.full((N,), action_dim, device=current_dist.logits.device) if head_infos[head_idx]["type"] == "categorical" else None
            # 自回归依赖调整
            if current_autoregressive and current_autoregressive[0] != -1:
                for dep_idx in current_autoregressive:
                    assert dep_idx < head_idx, \
                        f"动作头{current_head_name}依赖的索引{dep_idx}需小于当前索引{head_idx}"
                    dep_act = sampled_actions[dep_idx]
                    dep_act_val = dep_act.argmax(dim=-1)
                    available_acts = available_acts - dep_act_val
                    available_acts = torch.clamp(available_acts, min=1)

            # 生成掩码
            act_indices = torch.arange(action_dim, device=current_dist.logits.device) if head_infos[head_idx]["type"] == "categorical" else None
            mask = (act_indices < available_acts.unsqueeze(1)).float() if head_infos[head_idx]["type"] == "categorical" else None # (N, action_dim)
            # 保存掩码
            autoregressive_masks[current_head_name] = mask

            # 应用掩码并采样
            current_dist.set_mask(mask) if mask is not None else None
            current_act = current_dist.sample()
            sampled_actions.append(current_act)

        # 拼接多动作头动作
        action = torch.cat(sampled_actions, dim=-1)

        return action, autoregressive_masks
    
    def _split_action(self, action):
        """
        将拼接的动作张量按动作头配置拆分

        参数:
            action: 拼接后的动作张量，形状 (batch, total_action_dim)

        返回:
            split_actions: dict, {head_name: tensor(batch, action_dim_i)}
        """
        split_actions = {}
        current_dim = 0
        for head in self._head_infos:
            head_name = head["name"]
            out_dim = head["out_dim"]
            split_actions[head_name] = action[..., current_dim : current_dim + out_dim]
            current_dim += out_dim
        return split_actions