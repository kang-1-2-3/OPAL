import torch.utils.data as data
import numpy as np 
import pykitti 
import os 
from glob import glob
import yaml
from maploc.utils.geo import Projection
from maploc.osm.reader import OSMData, OSMNode, OSMWay
from pathlib import Path
from maploc.osm.tiling import TileManager, BoundaryBox
from gen_bev_images import getBEV
from PIL import Image
import torch
import cv2
import time 
import contextily as ctx
import tempfile
import matplotlib.pyplot as plt
import folium
import io
# from selenium import webdriver
from PIL import Image
import matplotlib.pyplot as plt
import contextily as ctx

class PcMapLocDataset(data.Dataset):
    def __init__(self, opt, mode):
        self.opt = opt
        # print(self.opt)
        # self.data_dir = data_dir
        # self.osm_data = self.load_osm_data()
        self.tile_size = self.opt['tiling']['tile_margin']
        self.data_list, self.tile_manager = self.make_dataset(mode)
        self.mode = mode
        # self.make_dataset()
    # def load_osm_data(self):
    #     osm_data = OSMData.from_file(Path(self.opt['loading']['osm_data_dir']))
    #     print(osm_data)
    #     return osm_data
    def make_dataset(self, mode = 'train'):
        if mode == 'train':
            # sequence_list = ["01", "04", "05", "06", "07", "08", "09", "10"]
            sequence_list = ['00']
        elif mode == 'val':
            sequence_list = ["00"]
        elif mode == 'test':
            sequence_list = ["00"]
        dataset = []
        tile_manager = {}
        # for seq in sequence_list:
        #     pc_gps_file_path = os.path.join(f"{self.opt['loading']['gps_data_dir']}", f"gps_sequence_{seq}.npy")
        #     assert os.path.exists(pc_gps_file_path)
        #     pc_gps_file = np.load(pc_gps_file_path, allow_pickle=True).item()
        #     # print(pc_gps_file.keys())
        #     kitti_dataset = pykitti.odometry(self.opt['loading']['pc_data_dir'], seq)
        #     # assert len(kitti_dataset) == len(pc_gps_file)
        #     if not os.path.exists(os.path.join(f"{self.opt['tiling']['tiles_path']}",f"80_tiles_{seq}.pkl")):
        #         self.save_seq_tile_manager(seq, pc_gps_file)
        #     seq_tile_manager = self.load_seq_tile_manager(seq)
        #     # add data to data_list
        #     for index, (pc_file_path, T_w_velo_i, lat, lon) in enumerate(zip(kitti_dataset.velo_files, pc_gps_file['T_w_velo_i'], pc_gps_file['lat'], pc_gps_file['lon'])):
        #         # pc_bev_img_path = os.path.join(f"{self.opt['loading']['pc_bev_dir']}", seq, f"{index:06d}.png")
        #         pc_bev_img_path = os.path.join(f"{self.opt['loading']['pc_bev_dir']}", seq, f"{index:06d}.png")
        #         dataset.append({'seq': seq, 
        #                         'index': index, 
        #                         'pc_file_path': pc_file_path, 'gps_trans_data': T_w_velo_i, 
        #                         'world_lla': pc_gps_file['world_lla'], 
        #                         'lat': lat, 
        #                         'lon': lon, 
        #                         'pc_bev_img_path': pc_bev_img_path})
        #     tile_manager[seq] = seq_tile_manager

        seq = '00'
        pc_gps_file_path = os.path.join(f"{self.opt['loading']['gps_data_dir']}", f"gps_sequence_{seq}.npy")
        assert os.path.exists(pc_gps_file_path)
        pc_gps_file = np.load(pc_gps_file_path, allow_pickle=True).item()
        # print(pc_gps_file.keys())
        kitti_dataset = pykitti.odometry(self.opt['loading']['pc_data_dir'], seq)
        # assert len(kitti_dataset) == len(pc_gps_file)
        if not os.path.exists(os.path.join(f"{self.opt['tiling']['tiles_path']}",f"80_tiles_{seq}.pkl")):
            self.save_seq_tile_manager(seq, pc_gps_file)
        seq_tile_manager = self.load_seq_tile_manager(seq)
        # add data to data_list
        if mode == 'train':
            frame_range = range(2000) 
        elif mode == 'val':
            frame_range = range(3000, 3500) 

        for index in frame_range:
            pc_file_path = kitti_dataset.velo_files[index]
            T_w_velo_i = pc_gps_file['T_w_velo_i'][index]
            lat = pc_gps_file['lat'][index]
            lon = pc_gps_file['lon'][index]
            
            pc_bev_img_path = os.path.join(f"{self.opt['loading']['pc_bev_dir']}", seq, f"{index:06d}.png")
            dataset.append({
                'seq': seq,
                'index': index,
                'pc_file_path': pc_file_path,
                'gps_trans_data': T_w_velo_i,
                'world_lla': pc_gps_file['world_lla'],
                'lat': lat,
                'lon': lon,
                'pc_bev_img_path': pc_bev_img_path
            })

        tile_manager[seq] = seq_tile_manager
        return dataset, tile_manager
    
    # def make_kitti_osm_pairs(self, sequence_list):
    #     dataset = []
    #     tile_manager = {}
    #     for seq in sequence_list:
            
    #         pc_gps_file_path = os.path.join(f"{self.opt['loading']['gps_data_dir']}", f"gps_sequence_{seq}.npy")
    #         assert os.path.exists(pc_gps_file_path)
    #         pc_gps_file = np.load(pc_gps_file_path, allow_pickle=True).item()
    #         # print(pc_gps_file.keys())
    #         kitti_dataset = pykitti.odometry(self.opt['loading']['pc_data_dir'], seq)
    #         # assert len(kitti_dataset) == len(pc_gps_file)
    #         if not os.path.exists(os.path.join(f"{self.opt['tiling']['tiles_path']}",f"tiles_{seq}.pkl")):
    #             self.save_seq_tile_manager(seq, pc_gps_file)
    #         seq_tile_manager = self.load_seq_tile_manager(seq)
    #         # add data to data_list
    #         for index, (pc_file_path, T_w_velo_i, lat, lon) in enumerate(zip(kitti_dataset.velo_files, pc_gps_file['T_w_velo_i'], pc_gps_file['lat'], pc_gps_file['lon'])):
    #             dataset.append({'seq': seq, 'index': index, 'pc_file_path': pc_file_path, 'gps_trans_data': T_w_velo_i, 'world_lla': pc_gps_file['world_lla'], 'lat': lat, 'lon': lon})
    #         tile_manager[seq] = seq_tile_manager
    #     # print(len(dataset))
    #     return dataset, tile_manager
    
    def save_seq_tile_manager(self, seq, pc_gps_file):
        seq_lat = pc_gps_file['lat']
        seq_lon = pc_gps_file['lon']
        tile_margin = self.opt['tiling']['tile_margin']
        seq_latlon = np.stack([seq_lat, seq_lon], 1)
        # print(seq_latlon.shape)
        projection = Projection.from_points(seq_latlon)
        seq_xy = projection.project(seq_latlon)

        bbox_map_min = np.floor(seq_xy.min(0) / tile_margin) * tile_margin 
        bbox_map_max = np.ceil(seq_xy.max(0) / tile_margin) * tile_margin
        seq_bbox = BoundaryBox(bbox_map_min, bbox_map_max)+tile_margin

        seq_tile_manager = TileManager.from_bbox(projection, seq_bbox, 2, tile_size = tile_margin)
        seq_tile_manager.save(os.path.join(self.opt['tiling']['tiles_path'], f"80_tiles_{seq}.pkl"))

    def load_seq_tile_manager(self, seq):
        tile_manager = TileManager.load(Path(os.path.join(self.opt['tiling']['tiles_path'], f"80_tiles_{seq}.pkl")))
        return tile_manager

    def pc_bev_generation(self, pcs: np.ndarray, seed: int):
        """
        Input pc shape --> [N, 3]
        Output pc_bev_img shape --> [H, W] (201, 201)

        With random rotation of the point cloud for data augmentation
        """
        np.random.seed(seed)
        ang = np.random.randint(360)/180.0*np.pi
        rot_mat = np.array([[np.cos(ang),np.sin(ang),0],[-np.sin(ang),np.cos(ang),0],[0,0,1]])
        # pcs = pcs.dot(rot_mat)
        pcs = pcs[np.where(np.abs(pcs[:,0])<40)[0],:]
        pcs = pcs[np.where(np.abs(pcs[:,1])<40)[0],:]
        pcs = pcs[np.where((np.abs(pcs[:,2])<40))[0],:]

        pcs = pcs.astype(np.float32)
        # print(pcs.shape)
        img, _, _ = getBEV(pcs)
        # print(1)
        return img
    def validate_seq_and_index(self,file_path, seq, seq_index):
        parts = file_path.split(os.sep)
        
        try:
            seq_from_path = parts[-3]  
            file_name = parts[-1]     
            seq_index_from_file = file_name.split('.')[0]  
            if seq == seq_from_path and seq_index == seq_index_from_file:
                return True
            else:
                return False
        except IndexError:
            return False
    def pad_or_trim_to_np(self, x, shape, pad_val=0):
        shape = np.asarray(shape)
        pad = shape - np.minimum(np.shape(x), shape)
        zeros = np.zeros_like(pad)
        x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
        return x[:shape[0], :shape[1]]
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        seed = index
        load_start = time.time()
        data_item = self.data_list[index]
        seq, seq_index, pc_file_path, gps_trans_data, world_lla, lat, lon, pc_bev_img_path = data_item['seq'], data_item['index'], data_item['pc_file_path'], data_item['gps_trans_data'], data_item['world_lla'], data_item['lat'], data_item['lon'], data_item['pc_bev_img_path']
        # print(lat, lon)
        # print(world_lla)
        # validate if the seq and seq_index correspond to the pc_file_path
        # self.validate_seq_and_index(pc_file_path, seq, seq_index)

        # PC BEV image generation
        
        """
        For pc visualization
        """
        # pcs = np.fromfile(pc_file_path,dtype=np.float32).reshape(-1,4)[:,:3] # [N, 3]
        # pcs = np.dot(gps_trans_data[:3,:3], pcs.T) + gps_trans_data[:3,3].reshape(-1, 1)
        # pcs = pcs.T
        # if index % 100 == 0:
        #     # print(f"Loading time: {time.time()-load_start}")
        #     plt.figure(figsize=(10, 10))
        #     plt.scatter(pcs[:, 0], pcs[:, 1], s=1, c='blue', alpha=0.5)
        #     plt.xlabel('X (m)')
        #     plt.ylabel('Y (m)')
        #     plt.title(f'Point Cloud Top View - Sequence {seq} Index {index}')
        #     plt.axis('equal') 
        #     plt.grid(True)
        #     plt.savefig(f'datavis/{seq}_{seq_index}_pc.png', dpi=300, bbox_inches='tight')
        #     plt.close()
        
        # pc_bev_img = self.pc_bev_generation(pcs, seed)/255.0 # [H, W]
        #  (201, 201)
        # cv2.imwrite(f'pc_bev_{seq}_{index}.png', pc_bev_img*255)

        # base_path = os.path.dirname(pc_bev_img_path)
        # path = os.path.basename(pc_bev_img_path).split('.')[0]
        # pc_bev_img_num = np.load(os.path.join(base_path, path+ '_0.npy'))
        # pc_bev_img_intensity = np.load(os.path.join(base_path, path+ '_1.npy'))
        # pc_bev_img_label = np.load(os.path.join(base_path, path+ '_2.npy'))
        # pc_bev_img = np.stack([pc_bev_img_num, pc_bev_img_intensity, pc_bev_img_label], 0)
        # pc_bev_img = pc_bev_img.transpose(1,2,0)
        
        # for pointpillar
        pcs = np.fromfile(pc_file_path,dtype=np.float32).reshape(-1,4) 
        label_path = pc_file_path.replace('/velodyne/', '/labels/').replace('.bin', '.label')
        label = np.fromfile(label_path, dtype=np.int32).reshape(-1)
        pcs = np.concatenate([pcs, label.reshape(-1,1)], axis=1)
        lidar_data = self.pad_or_trim_to_np(pcs, [81920, 5]).astype('float32')
        

        num_points = pcs.shape[0]
        lidar_mask = np.ones(81920).astype('float32')
        lidar_mask[num_points:] *= 0.0
        

        '''
        # for pc bev image
        pc_bev_img = cv2.imread(pc_bev_img_path)
        # cv2.imwrite(f"datavis/{seq}_{seq_index}_pc_bev_img.png", pc_bev_img)
        if self.mode == 'train':
            mat = cv2.getRotationMatrix2D((pc_bev_img.shape[1]//2, pc_bev_img.shape[0]//2 ), np.random.randint(0,360), 1)
            pc_bev_img = cv2.warpAffine(pc_bev_img, mat, pc_bev_img.shape[:2])
        pc_bev_img = pc_bev_img.transpose(2,0,1)
        # cv2.imwrite(f"pc_bev_img_{seq}_{seq_index}.png", pc_bev_img)
        '''
        seq_tile_manager = self.tile_manager[seq]
        latlon = np.array([lat, lon]) # [lat, lon] corresponds to the origin of point cloud in each frame. Here we transform it to the tile coordinate system
        proj = seq_tile_manager.projection
        xy = proj.project(latlon)
        # print(xy)
        sample_bbox = BoundaryBox(xy-self.tile_size//2, xy+self.tile_size//2)
        latlon_bbox = proj.unproject(sample_bbox)
        left_top = latlon_bbox.left_top
        right_bottom = latlon_bbox.right_bottom
        # print(left_top, right_bottom)
        
        # osm_image_path = f"osm_map_{seq}_{seq_index}.png"
        # self.get_osm_image(left_top, right_bottom, osm_image_path)
        
        canvas = seq_tile_manager.query(sample_bbox)
        """
        For OSM visualization
        """
        # from maploc.osm.viz import Colormap
        # vis = Colormap.apply(canvas.raster)
        # if index % 100 == 0:
        #     plt.imsave(f"datavis/{seq}_{seq_index}_canvas.png", vis.astype(np.float32))

        # TODO: Future work [Calculating overlap ratio between pc_bev_img and osm_map]
        # torch.from_numpy(np.ascontiguousarray(pc_bev_img)).long()
        return {
            'lidar_data': torch.from_numpy(lidar_data.astype(np.float32)), 
            'lidar_mask': torch.from_numpy(lidar_mask.astype(np.float32)),
            # 'pc_bev_img': torch.from_numpy(pc_bev_img.astype(np.float32)),
            'osm_map': torch.from_numpy(np.ascontiguousarray(canvas.raster)).long(),
            'xy': torch.from_numpy(xy.astype(np.float32))
        }

    def get_osm_image(self, left_top, right_bottom, output_path):
        """
        获取 OSM 底图并保存为 PNG 文件
        """
        bbox = (left_top[1], right_bottom[1], left_top[0], right_bottom[0])
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
        ctx.add_basemap(ax, crs="EPSG:4326", zoom=15, source=ctx.providers.CartoDB.Positron)
        ax.axis("off")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

if __name__ == "__main__":
    with open("conf/data/kitti.yaml", "r") as file:
        opt = yaml.safe_load(file)
    data_dir = opt['loading']['pc_data_dir']
    mode = 'val'
    dataset = PcMapLocDataset(opt, mode=mode)
    dataset[0]
    print(len(dataset))
    
    # for i in range(0, len(dataset), 100):
    #     data_item = dataset[i]
    #     print(i)
        # 处理 data_item
        # print(data_item['pc_bev_img'].shape, data_item['osm_map'].shape)
    # print(data1['pc_bev_img'].shape, data1['osm_map'].shape)
    # for i in range(0, len(dataset), 50):
    #     dataset[i]s