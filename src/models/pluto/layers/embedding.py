import torch
import torch.nn as nn
import torch.nn.functional as F
from natten import NeighborhoodAttention1D
from timm.models.layers import DropPath

# 实现了一个复杂的序列编码器（NATSequenceEncoder）以及相关的模块
# （如 ConvTokenizer, NATBlock, NATLayer, PointsEncoder 等）。这些模块主要用于处理一维序列数据（如时间序列、轨迹数据等），并通过注意力机制和卷积操作提取特征
# Neighborhood Attention Transformer(NAT)
# NOTE NATSequenceEncoder 是一个基于 Neighborhood Attention Transformer 的序列编码器，用于对输入的一维序列数据进行多层次的特征提取。
class NATSequenceEncoder(nn.Module):
    # in_chans=3: 输入序列的通道数（例如，时间序列的特征维度）。
    # embed_dim=32: 初始嵌入维度。
    # mlp_ratio=3: MLP 的隐藏层扩展比例。
    # kernel_size=[3, 3, 5]: 每一层的注意力窗口大小。
    # depths=[2, 2, 2]: 每一层的 Transformer 块数量。
    # num_heads=[2, 4, 8]: 每一层的多头注意力头数。
    # out_indices=[0, 1, 2]: 指定哪些层的输出需要被提取。
    # drop_rate=0.0: Dropout 概率。
    # attn_drop_rate=0.0: 注意力 Dropout 概率。
    # drop_path_rate=0.2: DropPath 概率。
    # norm_layer=nn.LayerNorm: 使用的归一化层。
    def __init__(
        self,
        in_chans=3,
        embed_dim=32,
        mlp_ratio=3,
        kernel_size=[3, 3, 5],
        depths=[2, 2, 2],
        num_heads=[2, 4, 8],
        out_indices=[0, 1, 2],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        # 使用 ConvTokenizer 将输入序列映射到 embed_dim 维度。
        self.embed = ConvTokenizer(in_chans, embed_dim)
        # NOTE 多层特征提取：self.levels 3层
        self.num_levels = len(depths)
        # NOTE 每一层的特征维度是 embed_dim * 2^i。
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_levels)]
        self.out_indices = out_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
        # NOTE 使用 NATBlock 作为每一层的特征提取模块。
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size[i],
                dilations=None,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
            )
            self.levels.append(level)

        # 归一化层：self.add_module
        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f"norm{i_layer}"
            # NOTE
            self.add_module(layer_name, layer)

        n = self.num_features[-1]
        # NOTE 特征融合：self.lateral_convs 和 self.fpn_conv
        self.lateral_convs = nn.ModuleList()
        for i_layer in self.out_indices:
            # padding=1:表示在输入数据的两端各填充 1 个单位的值（通常是 0），以保证卷积操作后输出的长度与输入的长度一致。
            # 输出长度 = (输入长度 + 2 * padding - 卷积核大小) / 步幅 + 1  步幅默认为1
            self.lateral_convs.append(
                nn.Conv1d(self.num_features[i_layer], n, 3, padding=1)
            )

        self.fpn_conv = nn.Conv1d(n, n, 3, padding=1)

    def forward(self, x):
        # B: 批量大小。
        # C: 通道数。
        # T: 时间步数。
        """x: [B, C, T]  ---> [B, T, C]"""
        # NOTE 使用 ConvTokenizer 将输入序列映射到嵌入空间。
        x = self.embed(x)
        # NOTE 多层特征提取
        # TAG step1: NATBlock
        out = []
        for idx, level in enumerate(self.levels):
            # NOTE 输出分别为下采样处理之后和原始输出
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                # 确保张量在内存中是连续的。在 PyTorch 中，某些操作（如 view）要求张量在内存中是连续的。
                # NOTE shape is (B, C, T)
                out.append(x_out.permute(0, 2, 1).contiguous())
        # TAG step2: conv1d
        # 使用 lateral_convs 对不同层的特征进行卷积。
        # 使用插值操作将高层次特征融合到低层次特征中。
        laterals = [
            lateral_conv(out[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(out) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                scale_factor=(laterals[i - 1].shape[-1] / laterals[i].shape[-1]),
                mode="linear",
                align_corners=False,
            )

        out = self.fpn_conv(laterals[0])
        # 使用 fpn_conv 进一步融合特征，并返回最后一个时间步的特征。
        return out[:, :, -1]


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=32, norm_layer=None):
        super().__init__()
        # NOTE
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # NOTE 使用卷积层提取特征，并将通道维度和时间维度交换。
        x = self.proj(x).permute(0, 2, 1)  # B, C, L -> B, L, C
        if self.norm is not None:
            x = self.norm(x)
        return x
    
# NOTE 适用场景
# 适用场景
# 时间序列处理：如轨迹预测、时间序列分类等。
# 特征提取：在深度网络中逐步减少时间步数，同时增加特征维度。
# Transformer 模型：作为下采样模块，用于减少序列长度。
# 通过卷积操作将输入序列的时间步数减半，同时将特征维度加倍。下采样后的特征会经过归一化处理（如 LayerNorm），以便后续网络更好地利用这些特征。
class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # NOTE dim: 输入特征的维度（即每个时间步的特征数）。即：通道数
        self.reduction = nn.Conv1d(
            dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        # NOTE 时间步数减半（stride=2）。特征维度加倍（dim -> 2 * dim）。
        # B T C ---> B C T --> B 2*C T/2 --> B T/2 2*C
        x = self.reduction(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x) # 归一化的作用是对每个时间步的特征进行标准化，提升训练的稳定性。
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# NOTE NATLayer 是一个基于 Neighborhood Attention 的模块，用于捕获局部注意力。
'''
这个模块的主要功能是：
使用局部注意力机制提取序列中的局部依赖关系。
使用 MLP 提取非线性特征。
通过残差连接和 DropPath 提升训练稳定性。
'''
class NATLayer(nn.Module):
    '''
dim: 输入特征的维度（即每个时间步的特征数）。
num_heads: 多头注意力的头数。
kernel_size=7: 局部注意力的窗口大小（即每次注意力操作覆盖的时间步数）。
dilation=None: 注意力窗口的扩张率（默认为无扩张）。
# NOTE mlp_ratio=4.0: MLP 的隐藏层扩展比例，隐藏层维度为 dim * mlp_ratio。
qkv_bias=True: 是否为查询（Q）、键（K）、值（V）添加偏置。
qk_scale=None: 查询和键的缩放因子（默认为 1 / sqrt(dim)）。
drop=0.0: Dropout 概率。
attn_drop=0.0: 注意力权重的 Dropout 概率。
drop_path=0.0: DropPath 概率，用于随机丢弃残差路径。
act_layer=nn.GELU: 激活函数，默认为 GELU。
norm_layer=nn.LayerNorm: 归一化层，默认为 LayerNorm。
    '''
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        # 使用 NeighborhoodAttention1D 实现局部注意力。
        '''
        NeighborhoodAttention1D 是一种局部注意力机制：
        # NOTE 它只关注局部窗口内的时间步（由 kernel_size 和 dilation 决定），从而降低计算复杂度。
        dim: 输入特征维度。
        kernel_size: 注意力窗口大小。
        dilation: 窗口扩张率，控制窗口内时间步的间隔。
        num_heads: 多头注意力的头数。
        qkv_bias: 是否为查询、键、值添加偏置。
        attn_drop: 注意力权重的 Dropout 概率。
        proj_drop: 输出的 Dropout 概率。
        '''
        self.attn = NeighborhoodAttention1D(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # DropPath 是一种正则化技术，用于随机丢弃残差路径，防止过拟合。
        # 如果 drop_path=0.0，则不使用 DropPath，直接返回输入。
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        # 使用 MLP 进一步提取特征。
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        shortcut = x
        '''
        对输入特征进行归一化（self.norm1）。
        使用局部注意力层（self.attn）提取局部依赖关系。
        将注意力输出与输入特征（shortcut）通过残差连接相加。
        如果 drop_path > 0.0，则随机丢弃部分残差路径。
        '''
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# NOTE NATBlock 是 NATSequenceEncoder 的核心模块，用于堆叠多个 NATLayer 并进行特征提取。
class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        # 下采样
        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            # 输入输出维度一样 因为残差连接保留了输入的形状
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x

# NOTE 使用pointNet类似的方式对矢量地图信息进行encoding refer from https://web.stanford.edu/~rqi/pointnet/
# PointsEncoder 的类，它是一个神经网络模块，用于对点集（如多边形的点）进行编码，提取全局特征
# PointsEncoder 的主要功能是对点集（如多边形的点）进行特征编码。输入是一个点集的特征矩阵（如点的位置、方向等），输出是一个全局特征向量，表示整个点集的特征。
class PointsEncoder(nn.Module):
    def __init__(self, feat_channel, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_mlp = nn.Sequential(
            nn.Linear(feat_channel, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.second_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.encoder_channel),
        )

    def forward(self, x, mask=None):
        """
        x:
        点集的特征矩阵，形状为 [B, M, C]：
        B: 批量大小（batch size）。
        M: 每个点集的点数量。
        C: 每个点的特征维度（feat_channel）。
        mask:
        一个布尔掩码，形状为 [B, M]，表示哪些点是有效的。
        值为 True 的位置表示该点有效，值为 False 表示该点无效。

        x : B M 3
        mask: B M
        -----------------
        feature_global : B C
        """

        bs, n, _ = x.shape
        device = x.device
        # x[mask]：从x中选择对应位置为true(也就有效位置)的数据 使用布尔掩码提取有效点的特征，形状为 [bt, num_valid_points, feat_channel]，其中 num_valid_points 是有效点的总数。
        x_valid = self.first_mlp(x[mask])  # B n 256
        x_features = torch.zeros(bs, n, 256, device=device)
        # x_features的shape不变
        # 初始化一个全零张量，形状为 [B, M, 256]，用于存储所有点的初步编码结果。
        # 将有效点的编码结果 x_valid 填充到对应的位置。
        x_features[mask] = x_valid
        # 对每个点集的点特征取最大值（沿点维度 M），得到全局特征。
        '''
            为什么取最大值？
            取最大值（max pooling）是一种常见的全局特征聚合方法，尤其在点云或点集处理任务中。它的作用是：

            提取全局特征：

            最大值操作能够捕获点集中最显著的特征，生成一个全局特征表示。
            例如，如果点集表示一个多边形，max pooling 可以提取该多边形的全局特征。
            对点的顺序不敏感：

            点集是无序的，max pooling 是一种无序不变的操作，能够忽略点的排列顺序。

            dim=1:

            表示沿着第 1 维（点的维度）取最大值。
            对于每个点集，取出所有点在每个特征维度上的最大值。
            返回值:

            torch.max 返回一个元组 (values, indices)：
            values: 每个特征维度的最大值，形状为 [B, C]。
            indices: 最大值对应的索引，形状为 [B, C]。
        '''
        # (B, C:256)
        pooled_feature = x_features.max(dim=1)[0]
        '''
            pooled_feature.unsqueeze(1):

            将全局特征的形状从 [B, 256] 扩展为 [B, 1, 256]。
            .repeat(1, n, 1):

            将全局特征复制 n 次，形状变为 [B, M, 256]。
            torch.cat([...], dim=-1):

            将点的初步编码结果 x_features 和全局特征拼接在一起，形状变为 [B, M, 512]。
        '''
        x_features = torch.cat(
            [x_features, pooled_feature.unsqueeze(1).repeat(1, n, 1)], dim=-1
        )

        x_features_valid = self.second_mlp(x_features[mask])
        res = torch.zeros(bs, n, self.encoder_channel, device=device)
        res[mask] = x_features_valid
        # [B, encoder_channel]。
        res = res.max(dim=1)[0]
        # PointsEncoder 的功能是对点集进行编码，提取全局特征。它结合了点的局部特征和全局特征，生成一个高维特征表示。
        return res
