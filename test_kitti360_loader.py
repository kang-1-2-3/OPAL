import torch.utils.data as data
import numpy as np 
import os 
import yaml
from maploc.utils.geo import Projection
from pathlib import Path
from maploc.osm.tiling import TileManager, BoundaryBox
import torch
import numba as nb


class PcDataset_Val(data.Dataset):
    def __init__(self, opt, mode, sequence_list):
        self.opt = opt
        self.tile_size = self.opt['tiling']['tile_margin']
        self.data_list, self.tile_manager = self.make_dataset(mode, sequence_list)
        self.mode = mode
        self.grid_size = np.asarray([480,360,32])
        self.ignore_label = 255
        with open('conf/semantic-kitti.yaml', 'r') as f:
            self.semantic_kitti = yaml.safe_load(f)

    def make_dataset(self, mode = 'val', sequence_list=None):
        dataset = []
        tile_manager = {}
        if mode == 'val':
            pc360_gps_file_path = os.path.join(f"{self.opt['loading']['gps_data_dir']}", f"kitti360_gps_data.npy")
            pc_gps_file = np.load(pc360_gps_file_path, allow_pickle=True).item()
            for seq in sequence_list:
                seq_tile_manager = self.load_seq_tile_manager(seq)
                for index in range(len(pc_gps_file[seq]['gps'])):
                    timestamp = pc_gps_file[seq]['timestamps'][index]
                    seq_file = os.path.join(self.opt['loading']['kitti360_data_dir'], seq, "velodyne_points/data", f"{int(timestamp):010d}.bin")
                    assert os.path.exists(seq_file)
                    # rotation = euler_to_rotation_matrix(pc_gps_file[seq]['gps'][index][3], pc_gps_file[seq]['gps'][index][4], pc_gps_file[seq]['gps'][index][5])
                    dataset.append({'seq': seq, 
                                    'index': index, 
                                    'pc_file_path': seq_file,
                                    'lat': pc_gps_file[seq]['gps'][index][0],
                                    'lon': pc_gps_file[seq]['gps'][index][1]
                                    })
                tile_manager[seq] = seq_tile_manager
        
        return dataset, tile_manager
    
    def save_seq_tile_manager(self, seq, pc_gps_file):
        seq_lat = pc_gps_file['lat']
        seq_lon = pc_gps_file['lon']
        tile_margin = self.opt['tiling']['tile_margin']
        seq_latlon = np.stack([seq_lat, seq_lon], 1)
        projection = Projection.from_points(seq_latlon)
        seq_xy = projection.project(seq_latlon)

        bbox_map_min = np.floor(seq_xy.min(0) / tile_margin) * tile_margin 
        bbox_map_max = np.ceil(seq_xy.max(0) / tile_margin) * tile_margin
        seq_bbox = BoundaryBox(bbox_map_min, bbox_map_max)+tile_margin

        seq_tile_manager = TileManager.from_bbox(projection, seq_bbox, 2, tile_size = tile_margin)
        seq_tile_manager.save(os.path.join(self.opt['tiling']['tiles_path'], f"100_tiles_test_{seq}.pkl"))

    def load_seq_tile_manager(self, seq):
        tile_manager = TileManager.load(Path(os.path.join(self.opt['tiling']['tiles_path'], f"100_tiles_test_{seq}.pkl")))
        return tile_manager

    def cart2polar(self, input_xyz):
        rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
        phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
        return np.stack((rho,phi,input_xyz[:,2]),axis=1)
    

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_item = self.data_list[index]
        seq, seq_index, pc_file_path,lat, lon = data_item['seq'], data_item['index'], data_item['pc_file_path'], data_item['lat'], data_item['lon']

        pcs = np.fromfile(pc_file_path,dtype=np.float32).reshape(-1,4)


        label_path = pc_file_path.replace(seq+'/velodyne_points/data', seq+'_pred').replace('.bin', '.label')
        raw_label = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
        inline_mask = (np.sqrt(pcs[:, 0]**2 + pcs[:, 1]**2) < 50) & (np.abs(pcs[:, 2]) < 3)
        pcs = pcs[inline_mask]
        raw_label = raw_label[inline_mask]
        learning_map = self.semantic_kitti['learning_map']
        map_func = np.vectorize(lambda x: learning_map.get(x))  
        labels = map_func(raw_label).reshape((-1,1))


        xyz_pol = self.cart2polar(pcs) 
        max_bound = np.asarray([50,np.pi,1.5])
        min_bound = np.asarray([3,-np.pi,-3])

        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1)
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int32)

        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,pcs[:,:2]),axis = 1)
        return_fea = return_xyz
        distance_feature_2d_original = compute_distance_feature_polar(xyz_pol)
        if self.mode != 'train':
            data_tuple = (grid_ind,labels,return_fea,index)
        else:
            data_tuple = (grid_ind,labels,return_fea)

        seq_tile_manager = self.tile_manager[seq]
        latlon = np.array([lat, lon])
        proj = seq_tile_manager.projection
        xy = proj.project(latlon)

        return {'data_tuple': data_tuple,
            'xy': torch.from_numpy(xy.astype(np.float32)),
            'pc_vis_mask': torch.from_numpy(distance_feature_2d_original.astype(np.float32))/50.0
        }

@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_label(processed_label,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i,3]] += 1
    processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

def collate_fn_BEV(data):
    grid_ind_stack = [d['data_tuple'][0] for d in data]
    point_label = [d['data_tuple'][1] for d in data]
    xyz = [d['data_tuple'][2] for d in data]
    
    xy =  torch.from_numpy(np.stack([d['xy'] for d in data]))
    pc_vis_mask = torch.from_numpy(np.stack([d['pc_vis_mask'] for d in data]))
    return grid_ind_stack,point_label,xyz, xy, pc_vis_mask

def compute_distance_feature_polar(xyz_pol, r_bins=480, phi_bins=360, r_min=3, r_max=50):
    phi_step = (2*np.pi) / phi_bins
    rspace = np.linspace(r_min, r_max, r_bins)
    mask = (xyz_pol[:, 0] >= r_min) & (xyz_pol[:, 0] <= r_max)
    rho_clipped = xyz_pol[mask, 0]
    phi_clipped = xyz_pol[mask, 1]
    phi_indices = ((phi_clipped + np.pi) / phi_step).astype(int)
    phi_indices = np.clip(phi_indices, 0, phi_bins - 1)
    max_rho_per_bin = np.zeros(phi_bins, dtype=np.float32)
    np.maximum.at(max_rho_per_bin, phi_indices, rho_clipped)
    distance_feature = (rspace[None] <= max_rho_per_bin[:, None])
    return distance_feature.T

def collate_fn_BEV_test(data):    
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),grid_ind_stack,point_label,xyz,index

import pandas as pd

class OSMDataset_Val(data.Dataset):
    def __init__(self, opt, mode, sequence_list):
        self.opt = opt
        self.tile_size = self.opt['tiling']['tile_margin']
        self.data_list, self.tile_manager = self.make_dataset(mode, sequence_list)
        self.mode = mode
        with open('conf/semantic-kitti.yaml', 'r') as f:
            self.semantic_kitti = yaml.safe_load(f)

    def make_dataset(self, mode = 'val', sequence_list=None):

        if sequence_list is None:
            sequence_list = ["2013_05_28_drive_0009_sync"]

        dataset = []
        tile_manager = {}
        if mode == 'val':
            for seq in sequence_list:
                osm_gps_file_path = os.path.join('pose_data', seq + '_latlon_osm_pose.csv')
                osm_gps_file = np.array(pd.read_csv(osm_gps_file_path))
                if not os.path.exists(os.path.join(f"{self.opt['tiling']['tiles_path']}",f"100_tiles_test_{seq}.pkl")):
                    self.save_seq_tile_manager(seq, osm_gps_file)
                seq_tile_manager = self.load_seq_tile_manager(seq)
        
                for index in range(len(osm_gps_file)):
                    lat = osm_gps_file[index][0]
                    lon = osm_gps_file[index][1]
                    dataset.append({'seq': seq, 
                                    'index': index, 
                                    'lat': lat, 
                                    'lon': lon})
                    tile_manager[seq] = seq_tile_manager
        
        return dataset, tile_manager
    
    def save_seq_tile_manager(self, seq, osm_gps_file):
        seq_lat = osm_gps_file[:,0]
        seq_lon = osm_gps_file[:,1]
        tile_margin = self.opt['tiling']['tile_margin']
        seq_latlon = np.stack([seq_lat, seq_lon], 1)
        projection = Projection.from_points(seq_latlon)
        seq_xy = projection.project(seq_latlon)

        bbox_map_min = np.floor(seq_xy.min(0) / tile_margin) * tile_margin 
        bbox_map_max = np.ceil(seq_xy.max(0) / tile_margin) * tile_margin
        seq_bbox = BoundaryBox(bbox_map_min, bbox_map_max)+tile_margin

        seq_tile_manager = TileManager.from_bbox(projection, seq_bbox, 2, tile_size = tile_margin, path=Path('data_osm/osm/karlsruhe.osm'))
        seq_tile_manager.save(os.path.join(self.opt['tiling']['tiles_path'], f"100_tiles_test_{seq}.pkl"))

    def load_seq_tile_manager(self, seq):
        tile_manager = TileManager.load(Path(os.path.join(self.opt['tiling']['tiles_path'], f"100_tiles_test_{seq}.pkl")))
        return tile_manager
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_item = self.data_list[index]
        seq, seq_index, lat, lon = data_item['seq'], data_item['index'], data_item['lat'], data_item['lon']
    
        seq_tile_manager = self.tile_manager[seq]
        latlon = np.array([lat, lon])
        proj = seq_tile_manager.projection
        xy = proj.project(latlon)

        raster_xy = xy
        sample_bbox = BoundaryBox(raster_xy-self.tile_size//2, raster_xy+self.tile_size//2)
        canvas = seq_tile_manager.query(sample_bbox)
        raster = canvas.raster

        return {        
            'osm_map': torch.from_numpy(np.ascontiguousarray(raster)).long(),
            'raster_xy': torch.from_numpy(raster_xy.astype(np.float32)),
        }

    
if __name__ == "__main__":
    with open("conf/data/kitti.yaml", "r") as file:
        opt = yaml.safe_load(file)
    data_dir = opt['loading']['pc_data_dir']
    mode = 'val'
    sequence_list = ["2013_05_28_drive_0009_sync"]
    dataset = OSMDataset_Val(opt, mode=mode, sequence_list=sequence_list)
    pc_dataset= PcDataset_Val(opt, mode=mode, sequence_list=sequence_list)
    pc_dataset[0]