import sys
sys.path.append('/data/Pcmaploc/code/geo-localization-with-point-clouds-and-openstreetmap')
from maploc.models.map_encoder_modified import MapEncoder   
import torch 
import torch.nn as nn
import torch.nn.functional as F
from BEVPlace.REIN import REIN, NetVLAD
import yaml
import numpy as np
from HDMapNet.model.homography import IPM
from HDMapNet.model.pointpillar import PointPillarEncoder
from HDMapNet.model.base import CamEncode
from HDMapNet.data.utils import gen_dx_bx
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from nuscenes_dataloader import NuScenesDataset

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)
    
class BevEncode(nn.Module):
    def __init__(self, inC, outC, instance_seg=False, embedded_dim=16, direction_pred=False, direction_dim=37):
        super(BevEncode, self).__init__()
        # trunk = resnet18(pretrained=False, zero_init_residual=True)
        trunk = resnet18(weights=None, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(64 + 256, 256, scale_factor=4)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )

        self.direction_pred = direction_pred
        if direction_pred:
            self.up1_direction = Up(64 + 256, 256, scale_factor=4)
            self.up2_direction = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, direction_dim, kernel_size=1, padding=0),
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)

        x = self.up1(x2, x1)
        x = self.up2(x)

        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1)
            x_embedded = self.up2_embedded(x_embedded)
        else:
            x_embedded = None

        if self.direction_pred:
            x_direction = self.up1_embedded(x2, x1)
            x_direction = self.up2_direction(x_direction)
        else:
            x_direction = None

        return x, x_embedded, x_direction


class ViewTransformation(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewTransformation, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
        self.hw_mat = nn.ModuleList(self.hw_mat)

    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H*W)
        outputs = []
        for i in range(N):
            output = self.hw_mat[i](feat[:, i]).view(B, C, self.bv_size[0], self.bv_size[1])
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs
    
class BEVPcMapLoc(nn.Module):
    def __init__(self, data_conf, embedded_dim=16):
        super(BEVPcMapLoc, self).__init__()

        # Map Encoder
        self.map_encoder = nn.Sequential(
            MapEncoder(data_conf['model']['map_encoder']),
            nn.MaxPool2d(kernel_size=5, stride=5)  
        )
        # Add NetVLAD layer for map features
        self.map_vlad = NetVLAD(num_clusters=32, dim=8)

        # Image and LiDAR Encoder
        self.camC = 64
        self.downsample = 16
        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        final_H, final_W = nx[1].item(), nx[0].item()

        self.camencode = CamEncode(self.camC)
        fv_size = (data_conf['image_size'][0]//self.downsample, data_conf['image_size'][1]//self.downsample)
        bv_size = (final_H//5, final_W//5)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)

        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=self.camC, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Lidar
        self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        self.bevencode = BevEncode(inC=self.camC+128, outC=data_conf['num_channels'], instance_seg=False, embedded_dim=embedded_dim, direction_pred=False, direction_dim=36+1)

        self.bev_vlad = NetVLAD(num_clusters=64, dim=16)
        # Add two 1x1 convolution layers to reduce bev_features_vlad from 1024 to 256
        self.reduce_conv1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.reduce_conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(256)
    
    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = None

        return Ks, RTs, post_RTs

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x
    
    def forward(self, map_data, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        # Encode the map data
        map_features = self.map_encoder[0](map_data)  # Pass through MapEncoder
        map_features_pooled = self.map_encoder[1](map_features['map_features']['feature_maps'][0])

        map_features_vlad = self.map_vlad(map_features_pooled) # [batch, 256]
        map_features_vlad = F.normalize(map_features_vlad, p=2, dim=1)
        # Encode the image data
        x = self.get_cam_feats(img)
        x = self.view_fusion(x)
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        topdown = self.up_sampler(topdown)

        # Encode the LiDAR data
        lidar_feature = self.pp(lidar_data, lidar_mask)
        topdown = torch.cat([topdown, lidar_feature], dim=1)
        bev_features, _, _ = self.bevencode(topdown) # [batch, 4, 200, 400]

        # NetVLAD for BEV features
        bev_features_vlad = self.bev_vlad(bev_features) # [batch, 1024]
        bev_features_vlad = F.relu(self.bn1(self.reduce_conv1(bev_features_vlad.unsqueeze(-1).unsqueeze(-1))))
        bev_features_vlad = F.relu(self.bn2(self.reduce_conv2(bev_features_vlad))).squeeze(-1).squeeze(-1) # [batch, 256]
        bev_features_vlad = F.normalize(bev_features_vlad, p=2, dim=1)
        return map_features_vlad, bev_features_vlad

if __name__ == '__main__':
    with open('conf/data/nuscenes.yaml', 'r') as file:
        conf = yaml.safe_load(file)

    device = conf['training']['device']
    model = BEVPcMapLoc(conf).to(device)
    train_dataset = NuScenesDataset(version='v1.0-trainval', dataroot='/data/Pcmaploc/data/Nuscenes', data_conf=conf, is_train=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)  

    first_batch = next(iter(train_loader))
    imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, osm_map, xy_torch = first_batch

    imgs = imgs.to(device)
    trans = trans.to(device)
    rots = rots.to(device)
    intrins = intrins.to(device)
    post_trans = post_trans.to(device)
    post_rots = post_rots.to(device)
    lidar_data = lidar_data.to(device)
    lidar_mask = lidar_mask.to(device)
    car_trans = car_trans.to(device)
    yaw_pitch_roll = yaw_pitch_roll.to(device)
    osm_data = osm_map.to(device)
    xy = xy_torch.to(device)

    osm_descs, pc_bev_descs = model(osm_data, imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll)
    # print(0)