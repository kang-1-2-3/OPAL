import torch
import torch.nn as nn
import torch_scatter
import numpy as np
# from .voxel import points_to_voxels

def raval_index(coords, dims):
    dims = torch.cat((dims, torch.ones(1, device=dims.device)), dim=0)[1:]
    dims = torch.flip(dims, dims=[0])
    dims = torch.cumprod(dims, dim=0) / dims[0]
    multiplier = torch.flip(dims, dims=[0])
    indices = torch.sum(coords * multiplier, dim=1)
    return indices

def points_to_voxels(
  points_xyz,
  points_mask,
  grid_range_x,
  grid_range_y,
  grid_range_z
):
    batch_size, num_points, _ = points_xyz.shape
    voxel_size_x = grid_range_x[2]
    voxel_size_y = grid_range_y[2]
    voxel_size_z = grid_range_z[2]
    grid_size = np.asarray([
        (grid_range_x[1]-grid_range_x[0]) / voxel_size_x,
        (grid_range_y[1]-grid_range_y[0]) / voxel_size_y,
        (grid_range_z[1]-grid_range_z[0]) / voxel_size_z
    ]).astype('int32')
    voxel_size = np.asarray([voxel_size_x, voxel_size_y, voxel_size_z])
    voxel_size = torch.Tensor(voxel_size).to(points_xyz.device)
    num_voxels = grid_size[0] * grid_size[1] * grid_size[2]
    grid_offset = torch.Tensor([grid_range_x[0], grid_range_y[0], grid_range_z[0]]).to(points_xyz.device)
    shifted_points_xyz = points_xyz - grid_offset
    voxel_xyz = shifted_points_xyz / voxel_size
    voxel_coords = voxel_xyz.int()
    grid_size = torch.from_numpy(grid_size).to(points_xyz.device)
    grid_size = grid_size.int()
    zeros = torch.zeros_like(grid_size)
    voxel_paddings = ((points_mask < 1.0) |
                      torch.any((voxel_coords >= grid_size) |
                                (voxel_coords < zeros), dim=-1))
    voxel_indices = raval_index(
      torch.reshape(voxel_coords, [batch_size * num_points, 3]), grid_size)
    voxel_indices = torch.reshape(voxel_indices, [batch_size, num_points])
    voxel_indices = torch.where(voxel_paddings,
                                torch.zeros_like(voxel_indices),
                                voxel_indices)
    voxel_centers = ((0.5 + voxel_coords.float()) * voxel_size + grid_offset)
    voxel_coords = torch.where(torch.unsqueeze(voxel_paddings, dim=-1),
                               torch.zeros_like(voxel_coords),
                               voxel_coords)
    voxel_xyz = torch.where(torch.unsqueeze(voxel_paddings, dim=-1),
                            torch.zeros_like(voxel_xyz),
                            voxel_xyz)
    voxel_paddings = voxel_paddings.float()

    voxel_indices = voxel_indices.long()
    points_per_voxel = torch_scatter.scatter_sum(
        torch.ones((batch_size, num_points), dtype=voxel_coords.dtype, device=voxel_coords.device) * (1-voxel_paddings),
        voxel_indices,
        dim=1,
        dim_size=num_voxels
    )

    voxel_point_count = torch.gather(points_per_voxel,
                                     dim=1,
                                     index=voxel_indices)


    voxel_centroids = torch_scatter.scatter_mean(
        points_xyz,
        voxel_indices,
        dim=1,
        dim_size=num_voxels)
    point_centroids = torch.gather(voxel_centroids, dim=1, index=torch.unsqueeze(voxel_indices, dim=-1).repeat(1, 1, 3))
    local_points_xyz = points_xyz - point_centroids

    result = {
        'local_points_xyz': local_points_xyz,
        'shifted_points_xyz': shifted_points_xyz,
        'point_centroids': point_centroids,
        'points_xyz': points_xyz,
        'grid_offset': grid_offset,
        'voxel_coords': voxel_coords,
        'voxel_centers': voxel_centers,
        'voxel_indices': voxel_indices,
        'voxel_paddings': voxel_paddings,
        'points_mask': 1 - voxel_paddings,
        'num_voxels': num_voxels,
        'grid_size': grid_size,
        'voxel_xyz': voxel_xyz,
        'voxel_size': voxel_size,
        'voxel_point_count': voxel_point_count,
        'points_per_voxel': points_per_voxel
    }


    return result

class PillarBlock(nn.Module):
  def __init__(self, idims=64, dims=64, num_layers=1,
               stride=1):
    super(PillarBlock, self).__init__()
    layers = []
    self.idims = idims
    self.stride = stride
    for i in range(num_layers):
      layers.append(nn.Conv2d(self.idims, dims, 3, stride=self.stride,
                              padding=1, bias=False))
      layers.append(nn.BatchNorm2d(dims))
      layers.append(nn.ReLU(inplace=True))
      self.idims = dims
      self.stride = 1
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    return self.layers(x)


class PointNet(nn.Module):
  def __init__(self, idims=64, odims=64):
    super(PointNet, self).__init__()
    self.pointnet = nn.Sequential(
      nn.Conv1d(idims, odims, kernel_size=1, bias=False),
      nn.BatchNorm1d(odims),
      nn.ReLU(inplace=True)
    )

  def forward(self, points_feature, points_mask):
    batch_size, num_points, num_dims = points_feature.shape
    points_feature = points_feature.permute(0, 2, 1)
    mask = points_mask.view(batch_size, 1, num_points)
    return self.pointnet(points_feature) * mask


class PointPillar(nn.Module):
  def __init__(self, C, xbound, ybound, zbound, embedded_dim=16, direction_dim=37):
    super(PointPillar, self).__init__()
    self.xbound = xbound
    self.ybound = ybound
    self.zbound = zbound
    self.embedded_dim = embedded_dim
    self.pn = PointNet(15, 64)
    self.block1 = PillarBlock(64, dims=64, num_layers=2, stride=1)
    self.block2 = PillarBlock(64, dims=128, num_layers=3, stride=2)
    self.block3 = PillarBlock(128, 256, num_layers=3, stride=2)
    self.up1 = nn.Sequential(
      nn.Conv2d(64, 64, 3, padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )
    self.up2 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
      nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )
    self.up3 = nn.Sequential(
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
      nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    self.conv_out = nn.Sequential(
      nn.Conv2d(448, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 128, 3, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, C, 1),
    )
    self.instance_conv_out = nn.Sequential(
      nn.Conv2d(448, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 128, 3, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, embedded_dim, 1),
    )
    self.direction_conv_out = nn.Sequential(
      nn.Conv2d(448, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 128, 3, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, direction_dim, 1),
    )

  def forward(self, points, points_mask,
    x, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll):
    points_xyz = points[:, :, :3]
    points_feature = points[:, :, 3:]
    voxels = points_to_voxels(
      points_xyz, points_mask, self.xbound, self.ybound, self.zbound
    )
    points_feature = torch.cat(
      [points, # 5
       torch.unsqueeze(voxels['voxel_point_count'], dim=-1), # 1
       voxels['local_points_xyz'], # 3
       voxels['point_centroids'], # 3
       points_xyz - voxels['voxel_centers'], # 3
      ], dim=-1
    )
    points_feature = self.pn(points_feature, voxels['points_mask'])
    voxel_feature = torch_scatter.scatter_mean(
      points_feature,
      torch.unsqueeze(voxels['voxel_indices'], dim=1),
      dim=2,
      dim_size=voxels['num_voxels'])
    batch_size = points.size(0)
    voxel_feature = voxel_feature.view(batch_size, -1, voxels['grid_size'][0], voxels['grid_size'][1])
    voxel_feature1 = self.block1(voxel_feature)
    voxel_feature2 = self.block2(voxel_feature1)
    voxel_feature3 = self.block3(voxel_feature2)
    voxel_feature1 = self.up1(voxel_feature1)
    voxel_feature2 = self.up2(voxel_feature2)
    voxel_feature3 = self.up3(voxel_feature3)
    voxel_feature = torch.cat([voxel_feature1, voxel_feature2, voxel_feature3], dim=1)
    return self.conv_out(voxel_feature).transpose(3, 2), self.instance_conv_out(voxel_feature).transpose(3, 2), self.direction_conv_out(voxel_feature).transpose(3, 2)


class PointPillarEncoder(nn.Module):
  def __init__(self, C, xbound, ybound, zbound):
    super(PointPillarEncoder, self).__init__()
    self.xbound = xbound
    self.ybound = ybound
    self.zbound = zbound
    self.pn = PointNet(15, 64)
    self.block1 = PillarBlock(64, dims=64, num_layers=2, stride=1)
    self.block2 = PillarBlock(64, dims=128, num_layers=3, stride=2)
    self.block3 = PillarBlock(128, 256, num_layers=3, stride=2)
    self.up1 = nn.Sequential(
      nn.Conv2d(64, 64, 3, padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )
    self.up2 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
      nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )
    self.up3 = nn.Sequential(
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
      nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    self.conv_out = nn.Sequential(
      nn.Conv2d(448, 256, 3, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 128, 3, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, C, 1),
    )

  def forward(self, points, points_mask):
    points_xyz = points[:, :, :3]
    points_feature = points[:, :, 3:]
    voxels = points_to_voxels(
      points_xyz, points_mask, self.xbound, self.ybound, self.zbound
    )
    points_feature = torch.cat(
      [points, # 5
       torch.unsqueeze(voxels['voxel_point_count'], dim=-1), # 1
       voxels['local_points_xyz'], # 3
       voxels['point_centroids'], # 3
       points_xyz - voxels['voxel_centers'], # 3
      ], dim=-1
    )
    points_feature = self.pn(points_feature, voxels['points_mask'])
    voxel_feature = torch_scatter.scatter_mean(
      points_feature,
      torch.unsqueeze(voxels['voxel_indices'], dim=1),
      dim=2,
      dim_size=voxels['num_voxels'])
    batch_size = points.size(0)
    voxel_feature = voxel_feature.view(batch_size, -1, voxels['grid_size'][0], voxels['grid_size'][1])
    voxel_feature1 = self.block1(voxel_feature)
    voxel_feature2 = self.block2(voxel_feature1)
    voxel_feature3 = self.block3(voxel_feature2)
    voxel_feature1 = self.up1(voxel_feature1)
    voxel_feature2 = self.up2(voxel_feature2)
    voxel_feature3 = self.up3(voxel_feature3)
    voxel_feature = torch.cat([voxel_feature1, voxel_feature2, voxel_feature3], dim=1)
    return self.conv_out(voxel_feature).transpose(3, 2)