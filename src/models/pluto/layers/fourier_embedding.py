# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# NOTE refer from https://github.com/ZikangZhou/QCNet/blob/main/layers/fourier_embedding.py
import math
from typing import List, Optional

import torch
import torch.nn as nn

# NOTE 用于将连续输入（如标量值）通过傅里叶特征和多层感知机（MLP）编码为高维特征表示。
'''
FourierEmbedding 是一种特征编码方法，常用于将连续值（如时间、位置、速度等）映射到高维特征空间。
它结合了傅里叶特征（Fourier Features）和神经网络的非线性变换，能够捕获输入的周期性和复杂特征。
'''
class FourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        '''
        hidden_dim:
        输出特征的维度（即编码后的高维特征的大小）。
        num_freq_bands:
        傅里叶频率的数量，表示使用多少个频率来生成傅里叶特征。

        一个嵌入层，用于为每个输入维度生成一组频率。
        嵌入层的权重矩阵形状为 [input_dim, num_freq_bands]，每一行表示一个输入维度的频率。
        '''
        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        # 定义 MLP 列表
        '''
            self.mlps:
            一个 ModuleList，包含多个 MLP（多层感知机），每个输入维度对应一个 MLP。
            每个 MLP 的结构为：
            输入层：线性层，将输入特征（傅里叶特征 + 原始值）映射到隐藏维度。
            归一化层：LayerNorm，对隐藏特征进行归一化。
            激活函数：ReLU，引入非线性。
            输出层：线性层，将隐藏特征映射到最终的隐藏维度。
            输入特征维度:
            每个 MLP 的输入特征维度为 num_freq_bands * 2 + 1：
            num_freq_bands * 2：傅里叶特征（余弦和正弦）。
            +1：原始输入值。
        '''
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
    # continuous_inputs:一个张量，表示连续输入变量，形状为 [batch_size, input_dim]
    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        '''
            continuous_inputs.unsqueeze(-1):
            将输入张量的最后一维扩展，形状从 [batch_size, input_dim] 变为 [batch_size, input_dim, 1]。
            self.freqs.weight:
            嵌入层的权重矩阵，形状为 [input_dim, num_freq_bands]。
            每个输入维度对应一组频率。
            * 2 * math.pi:
            将输入值与频率相乘，并乘以 2π，以生成傅里叶特征。
            x.cos() 和 x.sin():
            NOTE 分别计算余弦和正弦特征，形状为 [batch_size, input_dim, num_freq_bands]。
            torch.cat([...], dim=-1):
            将余弦特征、正弦特征和原始输入值拼接在一起，生成最终的傅里叶特征。
            NOTE 拼接后的形状为 [batch_size, input_dim, num_freq_bands * 2 + 1]。
        '''
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        '''
            1. List[Optional[torch.Tensor]]
            List[Optional[torch.Tensor]]：这是一个类型注解，表示 continuous_embs 是一个列表，列表中的每个元素可以是 torch.Tensor 或 None。
            2. [None] * self.input_dim
            [None]：一个包含单个 None 元素的列表。
            * self.input_dim：将这个列表重复 self.input_dim 次，生成一个长度为 self.input_dim 的列表，其中每个元素都是 None。
        '''
        continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
        for i in range(self.input_dim):
            # 提取 x 中第 i 个特征的所有数据。
            # NOTE shape is (bt, num_freq_bands * 2 + 1)--->[batch_size, hidden_dim]
            continuous_embs[i] = self.mlps[i](x[..., i, :])
        '''
            torch.stack(continuous_embs):
            将所有输入维度的编码结果堆叠在一起，形状为 [input_dim, batch_size, hidden_dim]。
            .sum(dim=0):
            对输入维度求和，聚合所有输入维度的特征，形状为 [batch_size, hidden_dim]。
        '''
        x = torch.stack(continuous_embs).sum(dim=0)
        return self.to_out(x)
