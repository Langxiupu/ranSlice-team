import torch
from torch import nn
from torch import distributions as torchd

import models
import networks
import tools

# 随即行为器
class Random(nn.Module):
    def __init__(self, config, act_space):
        super(Random, self).__init__()
        self._config = config
        self._act_space = act_space

    def actor(self, feat):
        if self._config.actor["dist"] == "onehot":
            # 这里是根据环境数量重复，然后根据设定的动作空间大小初始化张量
            return tools.OneHotDist(
                torch.zeros(
                    self._config.num_actions, device=self._config.device
                ).repeat(self._config.envs, 1)
            )
        else:
            # 如果不是离散分布那么直接使用均匀分布
            return torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(
                        self._act_space.low, device=self._config.device
                    ).repeat(self._config.envs, 1),
                    torch.tensor(
                        self._act_space.high, device=self._config.device
                    ).repeat(self._config.envs, 1),
                ),
                1,
            )

    def train(self, start, context, data):
        return None, {}


class Plan2Explore(nn.Module):
    def __init__(self, config, world_model, reward):
        super(Plan2Explore, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        self._reward = reward
        self._behavior = models.ImagBehavior(config, world_model)
        self.actor = self._behavior.actor
        # 计算特征大小（根据世界模型是否为离散动态特性）
        if config.dyn_discrete:
            # 特征大小 = 离散维度×离散数量 + 确定性维度
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            # 随机特征大小 = 离散维度×离散数量
            stoch = config.dyn_stoch * config.dyn_discrete
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
        # 根据配置的目标类型（disag_target）确定输入特征的维度
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.dyn_deter,
            "feat": config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]
        # 定义预测网络（MLP）的参数
        kw = dict(
            inp_dim=feat_size
            + (
                config.num_actions if config.disag_action_cond else 0
            ),  # pytorch version
            shape=size,
            layers=config.disag_layers,
            units=config.disag_units,
            act=config.act,
        )
        self._networks = nn.ModuleList(
            [networks.MLP(**kw) for _ in range(config.disag_models)]
        )
        # 定义优化器的参数
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "explorer",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw
        )

    def train(self, start, context, data):
        with tools.RequiresGrad(self._networks):
            metrics = {}
            stoch = start["stoch"] # 获取初始的随机状态
            # 展平除环境和时间步之外的所有维度
            if self._config.dyn_discrete:
                stoch = torch.reshape(
                    stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),))
                )
            target = {
                "embed": context["embed"],
                "stoch": stoch,
                "deter": start["deter"],
                "feat": context["feat"],
            }[self._config.disag_target]
            inputs = context["feat"]
            if self._config.disag_action_cond:
                # 若需要动作条件，则将动作与输入特征拼接
                inputs = torch.concat(
                    [inputs, torch.tensor(data["action"], device=self._config.device)],
                    -1,
                )
            metrics.update(self._train_ensemble(inputs, target))
        metrics.update(self._behavior._train(start, self._intrinsic_reward)[-1])
        return None, metrics

    def _intrinsic_reward(self, feat, state, action):
        inputs = feat
        if self._config.disag_action_cond:
            inputs = torch.concat([inputs, action], -1)
        # 所有预测网络对输入进行预测，并取mode，最可能的结果
        preds = torch.cat(
            [head(inputs, torch.float32).mode()[None] for head in self._networks], 0
        )
        # 预测结果的标准差的平均值（计算模型间的分歧）
        disag = torch.mean(torch.std(preds, 0), -1)[..., None]
        if self._config.disag_log:
            disag = torch.log(disag)
        # 计算内在奖励：分歧乘以探索缩放因子
        reward = self._config.expl_intr_scale * disag
        if self._config.expl_extr_scale:
            # 若启用外在奖励，将内在奖励与外在奖励相加
            reward += self._config.expl_extr_scale * self._reward(feat, state, action)
        return reward

    def _train_ensemble(self, inputs, targets):
        # 训练预测网络集合（最小化预测损失）
        with torch.cuda.amp.autocast(self._use_amp):
            if self._config.disag_offset:
                targets = targets[:, self._config.disag_offset :]
                inputs = inputs[:, : -self._config.disag_offset]
            targets = targets.detach()
            inputs = inputs.detach()
            # 所有预测网络对输入进行预测
            preds = [head(inputs) for head in self._networks]
            # 计算每个模型的对数似然，并取平均值
            likes = torch.cat(
                [torch.mean(pred.log_prob(targets))[None] for pred in preds], 0
            )

            loss = -torch.mean(likes)
        # 使用优化器更新预测网络参数，并返回训练指标
        metrics = self._expl_opt(loss, self._networks.parameters())
        return metrics
