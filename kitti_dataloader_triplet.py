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

        self.tile_size = self.opt['tiling']['tile_margin']
        self.data_list, self.tile_manager = self.make_dataset(mode)
        # self.data = []
        self.pairs = self.create_pairs()

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

    def create_pairs(self):
        pairs = []
        for i, data_item in enumerate(self.data_list):
            pos_samples = []
            neg_samples = []
            for j, candidate in enumerate(self.data_list):
                if i != j:
                    dist = np.linalg.norm(data_item['gps_trans_data'][:3, 3] - candidate['gps_trans_data'][:3, 3])
                    if dist < 5:
                        pos_samples.append(candidate)
                    elif dist > 7:
                        neg_samples.append(candidate)
            # Shuffle and select samples
            np.random.shuffle(pos_samples)
            np.random.shuffle(neg_samples)
            pos_samples = pos_samples[:1]
            neg_samples = neg_samples[:10]
            pairs.append({
                'anchor': data_item,
                'positive': pos_samples,
                'negative': neg_samples
            })
        return pairs

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_item = self.pairs[index]
        query_data, positive_data, negative_data = data_item['anchor'], data_item['positive'], data_item['negative']

        # load query
        pc_bev_img = cv2.imread(query_data['pc_bev_img_path'])
        mat = cv2.getRotationMatrix2D((pc_bev_img.shape[1]//2, pc_bev_img.shape[0]//2 ), np.random.randint(0,360), 1)
        pc_bev_img = cv2.warpAffine(pc_bev_img, mat, pc_bev_img.shape[:2])
        pc_bev_img = pc_bev_img.transpose(2,0,1)

        # load positive
        seq_tile_manager = self.tile_manager[positive_data[0]['seq']]
        latlon = np.array([positive_data[0]['lat'], positive_data[0]['lon']])
        proj = seq_tile_manager.projection
        xy = proj.project(latlon)
        sample_bbox = BoundaryBox(xy-self.tile_size//2, xy+self.tile_size//2)
        canvas = seq_tile_manager.query(sample_bbox)

        # load negative
        neg_osm_maps = []
        neg_xys = []
        for neg in negative_data:
            seq_tile_manager = self.tile_manager[neg['seq']]
            latlon = np.array([neg['lat'], neg['lon']])
            proj = seq_tile_manager.projection
            xy = proj.project(latlon)
            sample_bbox = BoundaryBox(xy-self.tile_size//2, xy+self.tile_size//2)
            canvas = seq_tile_manager.query(sample_bbox)
            neg_osm_maps.append(torch.from_numpy(np.ascontiguousarray(canvas.raster)).long())
            neg_xys.append(torch.from_numpy(xy.astype(np.float32)))

        neg_osm_maps = torch.stack(neg_osm_maps, 0) # [10 , 3, 160, 160]
        neg_xys = torch.stack(neg_xys, 0) # [10, 2]
        return {
            'pc_bev_img': torch.from_numpy(pc_bev_img.astype(np.float32)), 
            'osm_map': torch.from_numpy(np.ascontiguousarray(canvas.raster)).long(),
            'xy': torch.from_numpy(xy.astype(np.float32)),
            'neg_osm_maps': neg_osm_maps,
            'neg_xys': neg_xys
        }


if __name__ == "__main__":
    with open("conf/data/kitti.yaml", "r") as file:
        opt = yaml.safe_load(file)
    data_dir = opt['loading']['pc_data_dir']
    mode = 'val'
    dataset = PcMapLocDataset(opt, mode=mode)
    dataset[0]
    print(len(dataset))
    
    pairs = dataset.create_pairs()
    print(len(pairs))
    
    # for i in range(0, len(dataset), 100):
    #     data_item = dataset[i]
    #     print(i)
        # 处理 data_item
        # print(data_item['pc_bev_img'].shape, data_item['osm_map'].shape)
    # print(data1['pc_bev_img'].shape, data1['osm_map'].shape)
    # for i in range(0, len(dataset), 50):
    #     dataset[i]