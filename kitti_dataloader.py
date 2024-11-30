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

class PcMapLocDataset(data.Dataset):
    def __init__(self, opt, mode):
        self.opt = opt
        # print(self.opt)
        # self.data_dir = data_dir
        # self.osm_data = self.load_osm_data()
        self.tile_size = self.opt['tiling']['tile_margin']
        self.data_list, self.tile_manager = self.make_dataset(mode)
        self.data = []
        # self.make_dataset()
    # def load_osm_data(self):
    #     osm_data = OSMData.from_file(Path(self.opt['loading']['osm_data_dir']))
    #     print(osm_data)
    #     return osm_data
    def make_dataset(self, mode = 'train'):
        if mode == 'train':
            sequence_list = ["01", "04", "05", "06", "07", "08", "09", "10"]
        elif mode == 'val':
            sequence_list = ["02"]
        elif mode == 'test':
            sequence_list = ["00"]
        dataset = []
        tile_manager = {}
        for seq in sequence_list:
            pc_gps_file_path = os.path.join(f"{self.opt['loading']['gps_data_dir']}", f"gps_sequence_{seq}.npy")
            assert os.path.exists(pc_gps_file_path)
            pc_gps_file = np.load(pc_gps_file_path, allow_pickle=True).item()
            # print(pc_gps_file.keys())
            kitti_dataset = pykitti.odometry(self.opt['loading']['pc_data_dir'], seq)
            # assert len(kitti_dataset) == len(pc_gps_file)
            if not os.path.exists(os.path.join(f"{self.opt['tiling']['tiles_path']}",f"tiles_{seq}.pkl")):
                self.save_seq_tile_manager(seq, pc_gps_file)
            seq_tile_manager = self.load_seq_tile_manager(seq)
            # add data to data_list
            for index, (pc_file_path, T_w_velo_i, lat, lon) in enumerate(zip(kitti_dataset.velo_files, pc_gps_file['T_w_velo_i'], pc_gps_file['lat'], pc_gps_file['lon'])):
                dataset.append({'seq': seq, 'index': index, 'pc_file_path': pc_file_path, 'gps_trans_data': T_w_velo_i, 'world_lla': pc_gps_file['world_lla'], 'lat': lat, 'lon': lon})
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
        seq_tile_manager.save(os.path.join(self.opt['tiling']['tiles_path'], f"tiles_{seq}.pkl"))

    def load_seq_tile_manager(self, seq):
        tile_manager = TileManager.load(Path(os.path.join(self.opt['tiling']['tiles_path'], f"tiles_{seq}.pkl")))
        return tile_manager

    def pc_bev_generation(self, pcs: np.ndarray):
        """
        Input pc shape --> [N, 3]
        Output pc_bev_img shape --> [H, W] (201, 201)

        With random rotation of the point cloud for data augmentation
        """
        # print(pcs.shape)
        ang = np.random.randint(360)/180.0*np.pi
        rot_mat = np.array([[np.cos(ang),np.sin(ang),0],[-np.sin(ang),np.cos(ang),0],[0,0,1]])
        pcs = pcs.dot(rot_mat)
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
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_item = self.data_list[index]
        seq, seq_index, pc_file_path, gps_trans_data, world_lla, lat, lon = data_item['seq'], data_item['index'], data_item['pc_file_path'], data_item['gps_trans_data'], data_item['world_lla'], data_item['lat'], data_item['lon']
        # print(lat, lon)
        # print(world_lla)
        # validate if the seq and seq_index correspond to the pc_file_path
        self.validate_seq_and_index(pc_file_path, seq, seq_index)

        # PC BEV image generation
        pcs = np.fromfile(pc_file_path,dtype=np.float32).reshape(-1,4)[:,:3] # [N, 3]
        pc_bev_img = self.pc_bev_generation(pcs) # [H, W] (201, 201)
        # cv2.imwrite(f"pc_bev_img_{seq}_{seq_index}.png", pc_bev_img)
        # TODO: [Clip OSM map from seq_tile_manager with gps_data]
        seq_tile_manager = self.tile_manager[seq]
        # print(seq_tile_manager.bbox)

        # Data augmentation


        latlon = np.array([lat, lon]) # [lat, lon] corresponds to the origin of point cloud in each frame. Here we transform it to the tile coordinate system
        proj = seq_tile_manager.projection
        xy = proj.project(latlon)
        # print(xy)
        sample_bbox = BoundaryBox(xy-self.tile_size//2, xy+self.tile_size//2)
        canvas = seq_tile_manager.query(sample_bbox)
        # np.save(f"canvas_{seq}_{seq_index}.npy", canvas.raster)
        # print(canvas.raster.shape) # (3, 128, 128)


        # TODO: Future work [Calculating overlap ratio between pc_bev_img and osm_map]
        # torch.from_numpy(np.ascontiguousarray(pc_bev_img)).long()
        return {
            'pc_bev_img': torch.from_numpy(pc_bev_img.astype(np.float32)).repeat(3, 1, 1), 
            'osm_map': torch.from_numpy(np.ascontiguousarray(canvas.raster)).long()}


if __name__ == "__main__":
    with open("conf/data/kitti.yaml", "r") as file:
        opt = yaml.safe_load(file)
    data_dir = opt['loading']['pc_data_dir']
    mode = 'test'
    dataset = PcMapLocDataset(opt, mode=mode)

    print(len(dataset))
    # print(dataset[0])