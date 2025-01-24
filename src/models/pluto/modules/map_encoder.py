import torch
import torch.nn as nn

from ..layers.embedding import PointsEncoder
from ..layers.fourier_embedding import FourierEmbedding

# 对地图相关的多模态数据进行编码，提取特征
class MapEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
        use_lane_boundary=False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.use_lane_boundary = use_lane_boundary
        # 表示多边形的初始通道数（如点的位置、方向等）。
        self.polygon_channel = (
            polygon_channel + 4 if use_lane_boundary else polygon_channel
        )

        self.polygon_encoder = PointsEncoder(self.polygon_channel, dim)
        # 一个傅里叶嵌入层（FourierEmbedding），用于对速度限制进行编码。
        # 输入是标量（速度限制），输出是 dim 维的特征。
        self.speed_limit_emb = FourierEmbedding(1, dim, 64)
        # num_embeddings （int） – 嵌入字典的大小
        # embedding_dim （int） – 每个嵌入向量的大小
        # 一个嵌入层，用于对多边形的类型（如车道、建筑物等）进行编码。
        # 输入类别数为 3，输出特征维度为 dim。
        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)

    def forward(self, data) -> torch.Tensor:
        polygon_center = data["map"]["polygon_center"]
        polygon_type = data["map"]["polygon_type"].long()
        polygon_on_route = data["map"]["polygon_on_route"].long()
        polygon_tl_status = data["map"]["polygon_tl_status"].long()
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        polygon_speed_limit = data["map"]["polygon_speed_limit"]
        point_position = data["map"]["point_position"]
        point_vector = data["map"]["point_vector"]
        point_orientation = data["map"]["point_orientation"]
        valid_mask = data["map"]["valid_mask"]

        if self.use_lane_boundary:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                    point_position[:, :, 1] - point_position[:, :, 0],
                    point_position[:, :, 2] - point_position[:, :, 0],
                ],
                dim=-1,
            )
        else:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )
        # 批量大小 多边形数量 每个多边形的点数量 特征通道数。
        bs, M, P, C = polygon_feature.shape
        valid_mask = valid_mask.view(bs * M, P)
        polygon_feature = polygon_feature.reshape(bs * M, P, C)
        # bs, M, dim
        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, M, -1)

        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
            # unsqueeze(-1) 在最后一个维度上增加一个新的维度 最后一个维度是1
            polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        )
        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

        x_polygon += x_type + x_on_route + x_tl_status + x_speed_limit

        return x_polygon
