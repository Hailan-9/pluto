import torch
import torch.nn as nn

from ..layers.common_layers import build_mlp
from ..layers.embedding import NATSequenceEncoder

# AgentEncoder 是一个 PyTorch 模块（nn.Module），用于对车辆（包括自车和其他车辆）的历史状态进行编码。它将输入的历史轨迹数据（如位置、速度、方向等）转换为固定维度的特征表示。
class AgentEncoder(nn.Module):
    '''
    NOTE state_channel=6: 自车状态的输入通道数（如位置、速度、加速度等）。即自车状态输入的特征维度
    history_channel=9: 每个历史轨迹点的特征通道数（如位置变化、速度变化等）。
    dim=128: 最终嵌入特征的维度。
    hist_steps=21: 历史轨迹的时间步数。
    use_ego_history=False: 是否使用自车的历史轨迹。如果为 False，则只使用当前状态。
    drop_path=0.2: 丢弃路径率（用于正则化）。
    NOTE state_attn_encoder=True: 是否使用 StateAttentionEncoder 对自车状态进行编码。
    NOTE state_dropout=0.75: 自车状态编码器的丢弃率。
    '''
    def __init__(
        self,
        state_channel=6,
        history_channel=9,
        dim=128,
        hist_steps=21,
        use_ego_history=False,
        drop_path=0.2,
        state_attn_encoder=True,
        state_dropout=0.75,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.state_channel = state_channel
        # 仅仅使用自车当前的状态信息
        self.use_ego_history = use_ego_history
        self.hist_steps = hist_steps
        self.state_attn_encoder = state_attn_encoder
        # NOTE 他车的历史轨迹编码器：self.history_encoder
        # NATSequenceEncoder 是一个序列编码器，用于对他车的历史轨迹进行编码。
        # 输入通道数为 history_channel=9，输出嵌入维度为 dim // 4 = 32。

        self.history_encoder = NATSequenceEncoder(
            in_chans=history_channel, embed_dim=dim // 4, drop_path_rate=drop_path
        )
        # NOTE 自车状态编码器：self.ego_state_emb
        if not use_ego_history:
            if not self.state_attn_encoder:
                self.ego_state_emb = build_mlp(state_channel, [dim] * 2, norm="bn")
            else:
                self.ego_state_emb = StateAttentionEncoder(
                    state_channel, dim, state_dropout
                )
        # 定义了一个嵌入层，用于将车辆类别（如自车、其他车辆、行人等）编码为 dim=128 维的向量。
        self.type_emb = nn.Embedding(4, dim)
    # 静态方法 to_vector 将轨迹特征（如位置、速度等）转换为向量形式（如连续时间步之间的差分）。
    '''
    输入：
    feat: 输入特征张量（形状为 [batch_size, num_agents, time_steps, feature_dim]）。
    valid_mask: 有效性掩码，表示哪些时间步是有效的。
    NOTE 输出： 差分特征（如位置变化、速度变化）。
    '''
    @staticmethod
    def to_vector(feat, valid_mask):
        # 计算有效性掩码：只有连续的时间步都有效，差分才有效。
        # 只有当相邻的两个 valid_mask 值都是 True 时，vec_mask 中对应位置才是 True。
        # NOTE 张量或者numpy数组的切片操作
        vec_mask = valid_mask[..., :-1] & valid_mask[..., 1:]

        while len(vec_mask.shape) < len(feat.shape):
            # unsqueeze(-1) 会在张量的最后一个维度添加一维
            vec_mask = vec_mask.unsqueeze(-1)
        # NOTE torch.where(condition, x, y)：如果 condition 为 True，返回 x；否则返回 y。
        # NOTE 根据 vec_mask 的值：如果某一位置的掩码是 True，保留对应位置的差分结果；如果掩码是 False，将差分结果置为 0。 对应位置 对应位置 对应位置！！！！！！
        return torch.where(
            vec_mask,
            feat[:, :, 1:, ...] - feat[:, :, :-1, ...],
            torch.zeros_like(feat[:, :, 1:, ...]),
        )

    def forward(self, data):
        T = self.hist_steps
        '''
            data: 包含车辆的历史状态信息，通常是一个字典，包含以下字段：
            data["agent"]["position"]: 位置轨迹。
            data["agent"]["heading"]: 航向角轨迹。
            data["agent"]["velocity"]: 速度轨迹。
            data["agent"]["shape"]: 车辆形状。
            data["agent"]["category"]: 车辆类别。
            data["agent"]["valid_mask"]: 有效性掩码。
        '''
        # shape is (B, A, T)
        position = data["agent"]["position"][:, :, :T]
        heading = data["agent"]["heading"][:, :, :T]
        velocity = data["agent"]["velocity"][:, :, :T]
        shape = data["agent"]["shape"][:, :, :T]
        category = data["agent"]["category"].long()
        valid_mask = data["agent"]["valid_mask"][:, :, :T]
        # 使用 to_vector 方法计算差分特征（如位置变化、速度变化、方向变化等）。
        # 将所有特征拼接成一个张量。
        heading_vec = self.to_vector(heading, valid_mask)
        valid_mask_vec = valid_mask[..., 1:] & valid_mask[..., :-1]
        agent_feature = torch.cat(
            [
                self.to_vector(position, valid_mask),
                self.to_vector(velocity, valid_mask),
                torch.stack([heading_vec.cos(), heading_vec.sin()], dim=-1),
                shape[:, :, 1:],
                # 在最后一个维度上添加一个维度，并使用 unsqueeze(-1) 方法将其扩展为与 agent_feature 的最后一个维度相同。
                valid_mask_vec.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        # NOTE (batch_size, agents, timesteps， state_dim)
        bs, A, T, _ = agent_feature.shape
        agent_feature = agent_feature.view(bs * A, T, -1)
        '''
            valid_mask：一个布尔张量，形状为 (bs, A, T)，表示每个代理在每个时间步是否有效。
            NOTE valid_mask.any(-1)：沿着最后一个维度（时间步长 T）进行逻辑或操作，生成一个形状为 (bs, A) 的布尔张量。这个张量表示每个代理在任何时间步是否有效。
            flatten()：将生成的布尔张量展平为一个一维张量，形状为 (bs * A,)。
        '''
        valid_agent_mask = valid_mask.any(-1).flatten()
        # NOTE 使用 NATSequenceEncoder 对历史轨迹特征进行编码，生成固定维度的嵌入表示。
        # NOTE permute之后 shape变成了：（bs, state_dim, T）
        '''
            agent_feature[valid_agent_mask]索引后，agent_feature[valid_agent_mask] 的形状为 [M, T, C]，其中：
            M 是有效代理的总数（即 valid_agent_mask 中 True 的数量）。
            T 是时间步数。
            C 是特征维度。
            调用 .contiguous() 是为了确保张量在内存中是连续的。
            由于 permute 操作会改变张量的内存布局，某些后续操作（如 view 或 reshape）可能需要张量是连续的，因此需要调用 .contiguous()。
        '''
        x_agent_tmp = self.history_encoder(
            agent_feature[valid_agent_mask].permute(0, 2, 1).contiguous()
        )
        # 初始化一个张量 x_agent，形状为 [bs * A, self.dim]，用来存储所有代理的特征。
        x_agent = torch.zeros(bs * A, self.dim, device=position.device)
        '''
            NOTE x_agent[valid_agent_mask] 的形状为 [M, self.dim]，与 x_agent_tmp 的形状一致。
            NOTE 赋值操作只会将 x_agent_tmp 的值填充到 x_agent 中对应的有效位置，但不会改变 x_agent 的整体形状。
        '''
        x_agent[valid_agent_mask] = x_agent_tmp
        x_agent = x_agent.view(bs, A, self.dim)

        if not self.use_ego_history:
            ego_feature = data["current_state"][:, : self.state_channel]
            x_ego = self.ego_state_emb(ego_feature)
            x_agent[:, 0] = x_ego

        x_type = self.type_emb(category)

        return x_agent + x_type

# NOTE SDE模型；StateAttentionEncoder 是一个专门用于对自车状态进行编码的模块。
class StateAttentionEncoder(nn.Module):
    '''
    state_channel: 自车状态的输入通道数。也就是特征的维度数
    dim: 输出嵌入的维度。
    state_dropout=0.5: 状态丢弃率。
    '''
    def __init__(self, state_channel, dim, state_dropout=0.5) -> None:
        super().__init__()

        self.state_channel = state_channel
        self.state_dropout = state_dropout
        # 每个状态通道对应一个线性层，用于特征变换。
        self.linears = nn.ModuleList([nn.Linear(1, dim) for _ in range(state_channel)])
        # NOTE 多头自注意力机制，用于捕获状态之间的依赖关系。
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        # NOTE 可学习的位置编码。用于在 PyTorch 模型中定义一个可训练的位置嵌入（position embedding）参数。位置嵌入通常用于将位置信息注入到模型中，特别是在处理序列数据时。
        '''
            1. nn.Parameter
            nn.Parameter：PyTorch 中的一个类，用于定义可训练的参数。这些参数在模型的训练过程中会被自动更新。
            nn.Parameter 的实例可以被添加到模型的参数列表中，这样在调用优化器（如 torch.optim.Adam）时，这些参数会被自动更新。
            2. torch.Tensor(1, state_channel, dim)
            torch.Tensor(1, state_channel, dim)：创建一个形状为 (1, state_channel, dim) 的张量。这个张量的初始值是未初始化的，通常需要在后续进行初始化。
            1：表示批量大小（batch size）为 1。这通常用于位置嵌入，因为位置嵌入通常是共享的，不需要为每个样本单独学习。
            state_channel：表示状态通道数，即位置嵌入在第二个维度上的大小。
            dim：表示嵌入维度，即位置嵌入在第三个维度上的大小。
        '''
        self.pos_embed = nn.Parameter(torch.Tensor(1, state_channel, dim))
        # NOTE 通过定义一个可训练的查询参数，模型可以学习如何关注输入序列中的不同部分，从而提高性能。
        # NOTE 定义一个可训练的查询（query）参数。这个查询参数通常用于注意力机制中，特别是在多头注意力（Multi-Head Attention）或自注意力（Self-Attention）机制中,用于提取全局状态信息。
        '''
            创建张量：torch.Tensor(1, 1, dim) 创建一个未初始化的张量，形状为 (1, 1, dim)。
            定义可训练参数：nn.Parameter 将这个张量包装成一个可训练的参数，使其在模型的训练过程中可以被自动更新。
            添加到模型：self.query 将这个可训练的参数添加到模型的成员变量中，使其成为模型的一部分。
        '''
        self.query = nn.Parameter(torch.Tensor(1, 1, dim))
        # NOTE 使用正态分布初始化
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x):
        x_embed = []
        # 每个状态通道通过对应的线性层进行编码。
        for i, linear in enumerate(self.linears):
            x_embed.append(linear(x[:, i, None]))
        # NOTE
        x_embed = torch.stack(x_embed, dim=1)
        # 用于将一个位置嵌入（self.pos_embed）张量在批量维度（batch dimension）上重复多次，以匹配输入张量 x_embed 的批量大小。这样可以确保每个样本都有一个相同的位置嵌入
        '''
            repeat 方法用于在指定维度上重复张量。
            repeat(x_embed.shape[0], 1, 1)：在第一个维度（批量维度）上重复 x_embed.shape[0] 次，在第二个维度和第三个维度上重复 1 次。
            这确保了位置嵌入的形状从 (1, sequence_length, dim) 变为 (batch_size, sequence_length, dim)。
        '''
        pos_embed = self.pos_embed.repeat(x_embed.shape[0], 1, 1)
        x_embed += pos_embed
        '''
            这个属性在模型的 train() 和 eval() 方法中被自动设置：
            model.train()：将模型设置为训练模式，self.training 被设置为 True。
            model.eval()：将模型设置为评估模式，self.training 被设置为 False。
        '''
        if self.training and self.state_dropout > 0:
            # 可见，也就是使用的token
            visible_tokens = torch.zeros(
                (x_embed.shape[0], 3), device=x.device, dtype=torch.bool
            )
            # 丢弃的token
            '''
                dropout_tokens：创建一个形状为 (x_embed.shape[0], self.state_channel - 3) 的随机张量，
                然后将其转换为布尔张量，表示哪些 token 被丢弃。具体来说，如果随机值小于 self.state_dropout，则该 token 被丢弃。
            '''
            dropout_tokens = (
                torch.rand((x_embed.shape[0], self.state_channel - 3), device=x.device)
                < self.state_dropout
            )
            key_padding_mask = torch.concat([visible_tokens, dropout_tokens], dim=1)
        else:
            key_padding_mask = None

        query = self.query.repeat(x_embed.shape[0], 1, 1)
        '''
            它的输出通常是一个元组，形如 (attn_output, attn_weights)：

            attn_output: 注意力机制的输出，形状为 [B, T_query, D]。
            T_query 是查询的时间步数（在这里为 1）。
            attn_weights: 注意力权重，形状为 [B, num_heads, T_query, T_key]。
            表示查询向量与键向量之间的注意力分数。
            [0]:

            取出注意力机制的第一个输出，即 attn_output，形状为 [B, T_query, D]。
        '''
        '''
            注意力机制的输入参数
            query: 查询向量，用于提取序列的全局特征。

            通常是一个 learnable 参数（可学习的向量），形状为 [B, 1, D]，其中：
            B: 批量大小。
            1: 查询的时间步数（通常是 1，因为我们只需要一个全局特征）。
            D: 特征维度。
            查询向量的作用是告诉注意力机制“我们想要提取什么样的特征”。
            key=x_embed 和 value=x_embed:

            key: 键向量，表示序列中每个时间步的特征，用于与查询向量进行匹配。
            value: 值向量，表示序列中每个时间步的特征，用于生成最终的注意力输出。
            x_embed:
            输入序列的嵌入表示，形状为 [B, T, D]，其中：
            B: 批量大小。
            T: 时间步数（序列长度）。
            D: 特征维度。
            key_padding_mask=key_padding_mask:

            用于指示哪些时间步是有效的，哪些是填充的（padding）。
            key_padding_mask 的形状为 [B, T]，其中：
            值为 True 的位置表示该时间步是填充（padding），需要被忽略。
            值为 False 的位置表示该时间步是有效的。
        '''
        # NOTE 输出和query的shape是一样的
        x_state = self.attn(
            query=query,
            key=x_embed,
            value=x_embed,
            key_padding_mask=key_padding_mask,
        )[0]
        '''
            x_state:

            经过注意力机制计算后的输出，形状为 [B, T_query, D]。
            在这里，T_query=1，所以 x_state 的形状为 [B, 1, D]。
            x_state[:, 0]:

            提取第一个时间步的特征，形状为 [B, D]。
            由于查询向量只有一个时间步（T_query=1），所以这里提取的实际上是整个注意力机制的输出。
        '''
        return x_state[:, 0]
