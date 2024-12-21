import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_recur(model, std=np.sqrt(2), bias_const=0.0):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, std)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, bias_const)
        elif len(list(module.children())) > 0:
            layer_init_recur(module)


class Agent_ppo_minesweeper(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(2, 32, 3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(
                nn.Linear(64 * envs.single_observation_space.shape[1] * envs.single_observation_space.shape[2], 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action(self, x):
        return self.get_action_and_value(x)[0]

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)

        # logits_mu = torch.mean(logits, dim=1, keepdim=True)
        # logits_std = torch.std(logits, dim=1, keepdim=True)
        # bias = torch.flatten(x[:, 1], 1)
        # bias_mu = torch.mean(bias, dim=1, keepdim=True)
        # bias_std = torch.std(bias, dim=1, keepdim=True)
        # logits = logits - (logits_std * (bias - bias_mu) / (bias_std + 1e-7) + logits_mu)

        logits = logits - (torch.std(logits, dim=1, keepdim=True) * (
                    torch.flatten(x[:, 1], 1) - torch.mean(torch.flatten(x[:, 1], 1), dim=1, keepdim=True)) / (
                                       torch.std(torch.flatten(x[:, 1], 1), dim=1, keepdim=True) + 1e-7) + torch.mean(
            logits, dim=1, keepdim=True))

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class ConditionalBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        # 如果批量大小为1，不进行归一化
        if x.size(0) == 1:
            return x
        else:
            return super().forward(x)


# 替换网络中的 BatchNorm2d 层
def replace_bn_with_conditional(model):
    """
    卧槽这个太牛逼了！！！
    :param model: nn.Module
    :return: None
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, ConditionalBatchNorm2d(module.num_features, module.eps, module.momentum, module.affine,
                                                        module.track_running_stats))
        elif len(list(module.children())) > 0:  # 递归替换子模块
            replace_bn_with_conditional(module)


class Agent_ppo_minesweeper_mobilenet_v3_large(nn.Module):
    def __init__(self, envs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _mobilenet_v3_large = models.mobilenet_v3_large()
        _mobilenet_v3_large.features[0][0] = nn.Conv2d(2, 16, kernel_size=(3, 3),
                                                       stride=(2, 2), padding=(1, 1), bias=False)

        # 将_mobilenet_v3_large中的单卷积层和单线性层利用layer_init初始化
        layer_init_recur(_mobilenet_v3_large)

        # 替换 BatchNorm2d 层
        replace_bn_with_conditional(_mobilenet_v3_large)

        self.network = nn.Sequential(
            _mobilenet_v3_large.features,
            _mobilenet_v3_large.avgpool,
            nn.Flatten(),
        )

        self.actor = layer_init(nn.Linear(960, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(960, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action(self, x):
        return self.get_action_and_value(x)[0]

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)

        logits_mu = torch.mean(logits, dim=1, keepdim=True)
        logits_std = torch.std(logits, dim=1, keepdim=True)
        bias = torch.flatten(x[:, 1], 1)
        bias_mu = torch.mean(bias, dim=1, keepdim=True)
        bias_std = torch.std(bias, dim=1, keepdim=True)
        logits = logits - (logits_std * (bias - bias_mu) / (bias_std + 1e-7) + logits_mu)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class Agent_ppo_minesweeper_mobilenet_v3_small(nn.Module):
    def __init__(self, envs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _mobilenet_v3_small = models.mobilenet_v3_small()
        _mobilenet_v3_small.features[0][0] = nn.Conv2d(2, 16, kernel_size=(3, 3),
                                                       stride=(2, 2), padding=(1, 1), bias=False)

        # 将_mobilenet_v3_small中的单卷积层和单线性层利用layer_init初始化
        layer_init_recur(_mobilenet_v3_small)

        # 替换 BatchNorm2d 层
        replace_bn_with_conditional(_mobilenet_v3_small)

        self.network = nn.Sequential(
            _mobilenet_v3_small.features,
            _mobilenet_v3_small.avgpool,
            nn.Flatten(),
        )

        self.actor = layer_init(nn.Linear(576, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(576, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action(self, x):
        return self.get_action_and_value(x)[0]

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)

        logits_mu = torch.mean(logits, dim=1, keepdim=True)
        logits_std = torch.std(logits, dim=1, keepdim=True)
        bias = torch.flatten(x[:, 1], 1)
        bias_mu = torch.mean(bias, dim=1, keepdim=True)
        bias_std = torch.std(bias, dim=1, keepdim=True)
        logits = logits - (logits_std * (bias - bias_mu) / (bias_std + 1e-7) + logits_mu)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
