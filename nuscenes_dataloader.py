from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from typing import Any, Dict
from PIL import Image
from maploc.osm.tiling import TileManager, BoundaryBox
from maploc.utils.geo import Projection, TopocentricConverter
from nuscenes.utils.splits import create_splits_scenes
from HDMapNet.data.lidar import get_lidar_data
from HDMapNet.data.image import normalize_img, img_transform
from HDMapNet.data.utils import label_onehot_encoding
from HDMapNet.model.voxel import pad_or_trim_to_np
from HDMapNet.data.const import CAMS, NUM_CLASSES, IMG_ORIGIN_H, IMG_ORIGIN_W
from pathlib import Path
from matplotlib import pyplot as plt

class NuScenesDataset(Dataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(NuScenesDataset, self).__init__()
        self.is_train = is_train
        self.data_conf = data_conf
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.scenes = self.get_scenes(version, is_train)
        self.samples_with_map = self.get_samples_with_maps()
        self.converter_list = self.converters()
        from glob import glob
        if not glob(os.path.join('maploc/data/nuscenes', "*.osm")):
            self.init_tile_manager()
        self.tile_manager_list = self.load_tile_manager()
    def __len__(self):
        return len(self.samples_with_map)
    
    def get_scenes(self, version, is_train):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[version][is_train]

        return create_splits_scenes()[split]

    def get_samples_with_maps(self):
        samples_with_maps = []

        for samp in self.nusc.sample:
            scene = self.nusc.get('scene', samp['scene_token'])
            
            if scene['name'] not in self.scenes:
                continue

            log = self.nusc.get('log', scene['log_token'])
            map_name = log['location']  
            
            samples_with_maps.append((samp, map_name))

        samples_with_maps.sort(key=lambda x: (x[0]['scene_token'], x[0]['timestamp']))

        if self.is_train:
            return samples_with_maps[:3000]
        else:
            return samples_with_maps[:500]

    def get_lidar(self, rec):
        lidar_data = get_lidar_data(self.nusc, rec, nsweeps=3, min_distance=2.2)
        lidar_data = lidar_data.transpose(1, 0)
        num_points = lidar_data.shape[0]
        lidar_data = pad_or_trim_to_np(lidar_data, [81920, 5]).astype('float32')
        lidar_mask = np.ones(81920).astype('float32')
        lidar_mask[num_points:] *= 0.0
        return lidar_data, lidar_mask

    def get_ego_pose(self, rec):
        sample_data_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        car_trans = ego_pose['translation']
        pos_rotation = Quaternion(ego_pose['rotation'])
        yaw_pitch_roll = pos_rotation.yaw_pitch_roll
        return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)

    def sample_augmentation(self):
        fH, fW = self.data_conf['image_size']
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)
        resize_dims = (fW, fH)
        return resize, resize_dims
    def get_imgs(self, rec):
        imgs = []
        trans = []
        rots = []
        intrins = []
        post_trans = []
        post_rots = []

        for cam in CAMS:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)

            resize, resize_dims = self.sample_augmentation()
            img, post_rot, post_tran = img_transform(img, resize, resize_dims)
            # resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            # img, post_rot, post_tran = img_transform(img, resize, resize_dims, crop, flip, rotate)

            img = normalize_img(img)
            post_trans.append(post_tran)
            post_rots.append(post_rot)
            imgs.append(img)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            trans.append(torch.Tensor(sens['translation']))
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))
            intrins.append(torch.Tensor(sens['camera_intrinsic']))
        return torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(post_trans), torch.stack(post_rots)

    def converters(self):
        origins = {
            "boston_seaport": [42.336849169438615, -71.05785369873047, 0],
            "singapore_onenorth": [1.2882100868743724, 103.78475189208984, 0],
            "singapore_hollandvillage": [1.2993652317780957, 103.78217697143555, 0],
            "singapore_queenstown": [1.2782562240223188, 103.76741409301758, 0],
        }

        converters = {name: TopocentricConverter(*origin) for name, origin in origins.items()}
        return converters
    
    def init_tile_manager(self):
        converters = self.converters()
        
        map_poses = {}
        for sample_with_map in self.samples_with_map:
            sample = sample_with_map[0]
            map_name = sample_with_map[1].replace("-", "_")
            
            sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            local_pose = ego_pose['translation'][:2]
            
            converter = converters[map_name]
            global_pose = converter.to_lla(local_pose[0], local_pose[1], 0)[:2] 
            
            if map_name not in map_poses:
                map_poses[map_name] = []
            map_poses[map_name].append(global_pose)
            
        tile_margin = 80  
        for map_name, poses in map_poses.items():
            poses = np.array(poses)
            projection = Projection.from_points(poses)
            poses_xy = projection.project(poses)
            
            bbox_map_min = np.floor(poses_xy.min(0) / tile_margin) * tile_margin
            bbox_map_max = np.ceil(poses_xy.max(0) / tile_margin) * tile_margin
            bbox = BoundaryBox(bbox_map_min, bbox_map_max) + tile_margin
            
            # bbox_osm = projection.unproject(bbox)
            # print(f"{map_name}, {bbox_osm}")
            tile_manager = TileManager.from_bbox(projection, bbox, 2, path=Path(f'maploc/data/nuscenes/{map_name}.osm'),tile_size=tile_margin)
            
            save_path = os.path.join("maploc/data/nuscenes", f"tiles_{map_name}.pkl")  
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            tile_manager.save(save_path)
            print(map_name, "saved")

    def load_tile_manager(self):
        map_names = ["boston_seaport", "singapore_onenorth", "singapore_hollandvillage", "singapore_queenstown"]
        tile_manager_list = []
        for map_name in map_names:
            path = os.path.join("maploc/data/nuscenes", f"tiles_{map_name}.pkl")
            tile_manager = TileManager.load(Path(path))
            tile_manager_list.append(tile_manager)

        return tile_manager_list
    def __getitem__(self, idx):

        rec = self.samples_with_map[idx][0]
        # print(rec)
        map_name = self.samples_with_map[idx][1]

        map_name = map_name.replace("-", "_")
        tile_manager = self.tile_manager_list[["boston_seaport", "singapore_onenorth", "singapore_hollandvillage", "singapore_queenstown"].index(map_name)]
        converter = self.converter_list[map_name]

        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        
        ## for pc vis
        # sample_data_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        # ego_pose = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        # car_trans_1 = ego_pose['translation']
        # pos_rotation = Quaternion(ego_pose['rotation'])
        # car_trans_1 = np.asarray(car_trans_1, dtype=np.float32)
        # rotation_matrix = pos_rotation.rotation_matrix

        # pcs = lidar_data[:, :3]
        # pcs = pcs @ rotation_matrix.T + car_trans_1

        # if idx % 100 == 0:
        #     plt.figure(figsize=(10, 10))
        #     plt.scatter(lidar_data[:, 0], lidar_data[:, 1], s=1, c='blue', alpha=0.5)
        #     plt.xlabel('X (m)')
        #     plt.ylabel('Y (m)')
        #     plt.title(f'Point Cloud Top View')
        #     plt.axis('equal') 
        #     plt.grid(True)
        #     plt.savefig(f'{idx}_pc.png', dpi=300, bbox_inches='tight')
        #     plt.close()

        car_trans_xy = car_trans[:2].numpy()
        latlon = converter.to_lla(car_trans_xy[0], car_trans_xy[1], 0)
        proj = tile_manager.projection
        xy = proj.project(latlon)

        sample_bbox = BoundaryBox(xy-80//2, xy+80//2)
        canvas = tile_manager.query(sample_bbox)
        osm_map = torch.from_numpy(np.ascontiguousarray(canvas.raster)).long()
        xy_torch = torch.from_numpy(xy.astype(np.float32))
        ## for map vis
        # from maploc.osm.viz import Colormap
        # vis = Colormap.apply(canvas.raster)
        
        # if idx % 100 == 0:
        #     plt.imsave(f"datavis/nuscenes/{idx}_canvas.png", vis.astype(np.float32))
        #     print(f"{idx} finished")
        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, osm_map, xy_torch


if __name__ == "__main__":
    data_conf = {
        'image_size': (900, 1600),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }
    train_dataset = NuScenesDataset(version='v1.0-trainval', dataroot='/data/Pcmaploc/data/Nuscenes', data_conf=data_conf, is_train=True)
    print(len(train_dataset))
    # for idx in range(0, train_dataset.__len__(), 100):
    #     imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, canvas = train_dataset.__getitem__(idx)
    train_dataset[18000]