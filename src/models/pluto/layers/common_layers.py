import torch.nn as nn
# NOTE 快速构建MLP网络的模块(py文件)
# MLP 是一种常见的神经网络结构，由多个全连接层（nn.Linear）组成，通常用于特征提取或分类任务。
# 这个函数的作用是根据输入参数动态构建一个 MLP 网络，支持可选的正则化（如 BatchNorm）和激活函数（如 ReLU）。

# c_in: 输入特征的维度（即每一层的输入通道数）。
# channels: 一个列表，定义每一层的输出通道数。例如，channels=[128, 64, 32] 表示 MLP 有 3 层，输出通道数分别为 128、64 和 32。
def build_mlp(c_in, channels, norm=None, activation="relu"):
    layers = [] # 用于存储 MLP 的每一层（包括全连接层、正则化层和激活函数）。
    # mlp的层数
    num_layers = len(channels)

    if norm is not None:
        norm = get_norm(norm)

    activation = get_activation(activation)

    for k in range(num_layers):
        if k == num_layers - 1:
            layers.append(nn.Linear(c_in, channels[k], bias=True))
        else:
            if norm is None:
                layers.extend([nn.Linear(c_in, channels[k], bias=True), activation()])
            else:
                layers.extend(
                    [
                        nn.Linear(c_in, channels[k], bias=False),
                        norm(channels[k]),
                        activation(),
                    ]
                )
            c_in = channels[k]
    # 将所有层（存储在 layers 列表中）打包成一个 nn.Sequential 模块，返回一个完整的 MLP 网络。
    # 使用星号 * 进行解包操作，将 layers 列表中的所有元素作为独立的参数传递给 nn.Sequential 构造函数。
    return nn.Sequential(*layers)


def get_norm(norm: str):
    if norm == "bn":
        return nn.BatchNorm1d
    elif norm == "ln":
        return nn.LayerNorm
    else:
        raise NotImplementedError


def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    else:
        raise NotImplementedError
