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
# from gen_bev_images import getBEV
import torch
import numba as nb

class PcDataset_Val(data.Dataset):
    def __init__(self, opt, seq, mode,rot = -1, if_remove_mov=False):
        self.opt = opt
        self.tile_size = self.opt['tiling']['tile_margin']
        self.seq =seq
        self.data_list, self.tile_manager = self.make_dataset(mode)
        self.mode = mode
        self.grid_size = np.asarray([480,360,32])
        self.ignore_label = 255
        self.rot = rot
        self.if_remove_mov = if_remove_mov
        with open('conf/semantic-kitti.yaml', 'r') as f:
            self.semantic_kitti = yaml.safe_load(f)
        # self.make_dataset()
        
    def make_dataset(self, mode = 'val'):
        if mode == 'train':
            sequence_list = ["01","02", "04", "05", "06", "08", "09", "10"]
            # sequence_list = ['00']
        elif mode == 'val':
            sequence_list = [self.seq]
        elif mode == 'test':
            sequence_list = ["00"]
        dataset = []
        tile_manager = {}
        for seq in sequence_list:
            pc_gps_file_path = os.path.join(f"{self.opt['loading']['gps_data_dir']}", f"gps_sequence_{seq}.npy")
            assert os.path.exists(pc_gps_file_path)
            pc_gps_file = np.load(pc_gps_file_path, allow_pickle=True).item()
            kitti_dataset = pykitti.odometry(self.opt['loading']['pc_data_dir'], seq)
            if not os.path.exists(os.path.join(f"{self.opt['tiling']['tiles_path']}",f"100_tiles_test_{seq}.pkl")):
                self.save_seq_tile_manager(seq, pc_gps_file)
            seq_tile_manager = self.load_seq_tile_manager(seq)
            # add data to data_list
            for index, (pc_file_path, T_w_velo_i, lat, lon) in enumerate(zip(kitti_dataset.velo_files, pc_gps_file['T_w_velo_i'], pc_gps_file['lat'], pc_gps_file['lon'])):
                dataset.append({'seq': seq, 
                                'index': index, 
                                'pc_file_path': pc_file_path, 'gps_trans_data': T_w_velo_i, 
                                'lat': lat, 
                                'lon': lon})
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

        seq_tile_manager = TileManager.from_bbox(projection, seq_bbox, 2, tile_size = tile_margin, path=Path('data_osm/osm/karlsruhe.osm'))
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
    
    def augment_point_cloud_with_2d_rotation(self, point_cloud,theta = None):
        x, y = point_cloud[:, 0], point_cloud[:, 1]
        if theta is None:
            theta = np.random.uniform(0, 2 * np.pi)
        else:
            theta = theta / 180 * np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_xy = np.dot(np.column_stack((x, y)), rotation_matrix.T)
        point_cloud[:, 0] = rotated_xy[:, 0]
        point_cloud[:, 1] = rotated_xy[:, 1]
        return point_cloud

    def __getitem__(self, index):
        data_item = self.data_list[index]
        seq, seq_index, pc_file_path, gps_trans_data, lat, lon = data_item['seq'], data_item['index'], data_item['pc_file_path'], data_item['gps_trans_data'], data_item['lat'], data_item['lon']

        pcs = np.fromfile(pc_file_path,dtype=np.float32).reshape(-1,4)
        if self.rot > 0:
            pcs = self.augment_point_cloud_with_2d_rotation(pcs,None) # random rot

        # gt_label_path = pc_file_path.replace('/velodyne/', '/labels/').replace('.bin', '.label')
        label_path = pc_file_path.replace('/velodyne/', '/pred_labels/').replace('.bin', '.label')
        # label_path = pc_file_path.replace('/velodyne/', '/predictions/').replace('.bin', '.label') # rangenet
        raw_label = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
        inline_mask = (np.sqrt(pcs[:, 0]**2 + pcs[:, 1]**2) < 50) & (np.abs(pcs[:, 2]) < 3)
        pcs = pcs[inline_mask]
        raw_label = raw_label[inline_mask]
        # if self.if_remove_mov:

        #     gt_label = np.fromfile(gt_label_path, dtype=np.uint32) & 0xFFFF
        #     gt_label = gt_label[inline_mask]

            # moving_mask = np.isin(gt_label, [252, 253, 254, 255, 256, 257, 258, 259])
            # keep_mask = ~moving_mask

            # apply mask in raw pcs and labels
            # pcs = pcs[keep_mask]
            # raw_label = raw_label[keep_mask]

        learning_map = self.semantic_kitti['learning_map']
        map_func = np.vectorize(lambda x: learning_map.get(x))  
        labels = map_func(raw_label).reshape((-1,1))

        xyz_pol = self.cart2polar(pcs) 
        max_bound = np.asarray([50,np.pi,1.5])
        min_bound = np.asarray([3,-np.pi,-3])

        # get grid index
        crop_range = max_bound - min_bound # [47, 2pi, 4.5]
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1) # [0.09812108559498957, 0.01750190893364787, 0.14516129032258066]
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int32) # [N, 3]

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,pcs[:,:2]),axis = 1)
        return_fea = return_xyz
        distance_feature_2d_original = compute_distance_feature_polar(xyz_pol)
        if self.mode != 'train':
            data_tuple = (grid_ind,labels,return_fea,index)
        else:
            data_tuple = (grid_ind,labels,return_fea)

        # For OSM Tile
        seq_tile_manager = self.tile_manager[seq]
        latlon = np.array([lat, lon]) # [lat, lon] corresponds to the origin of point cloud in each frame. Here we transform it to the tile coordinate system
        proj = seq_tile_manager.projection
        xy = proj.project(latlon)

        return {'data_tuple': data_tuple,
            'xy': torch.from_numpy(xy.astype(np.float32)),
            'pc_vis_mask': torch.from_numpy(distance_feature_2d_original.astype(np.float32)) /50.0
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
    # data2stack = np.stack([d['data_tuple'][0] for d in data]).astype(np.float32)
    # label2stack = np.stack([d['data_tuple'][1] for d in data])
    grid_ind_stack = [d['data_tuple'][0] for d in data]
    point_label = [d['data_tuple'][1] for d in data]
    xyz = [d['data_tuple'][2] for d in data]
    
    # osm_map = torch.from_numpy(np.stack([d['osm_map'] for d in data]))
    xy =  torch.from_numpy(np.stack([d['xy'] for d in data]))
    pc_vis_mask = torch.from_numpy(np.stack([d['pc_vis_mask'] for d in data]))
    # trans_pc_osm = torch.from_numpy(np.stack([d['trans_pc_osm'] for d in data]))
    # raster_xy = torch.from_numpy(np.stack([d['raster_xy'] for d in data]))
    return grid_ind_stack,point_label,xyz, xy, pc_vis_mask


def compute_distance_feature_polar(xyz_pol, r_bins=480, phi_bins=360, r_min=3, r_max=50):
    """
    Use NumPy's vectorization approach to directly compute distance features based on polar coordinates xyz_pol.
    xyz_pol: [N, 3], (rho, phi, z)。
    """
    phi_step = (2*np.pi) / phi_bins
    rspace = np.linspace(r_min, r_max, r_bins)
    # filter out points out of range
    mask = (xyz_pol[:, 0] >= r_min) & (xyz_pol[:, 0] <= r_max)
    rho_clipped = xyz_pol[mask, 0]
    phi_clipped = xyz_pol[mask, 1]
    # calculate the indice for each point
    phi_indices = ((phi_clipped + np.pi) / phi_step).astype(int)
    phi_indices = np.clip(phi_indices, 0, phi_bins - 1)
    # maximum rho (visible area) in each bi
    max_rho_per_bin = np.zeros(phi_bins, dtype=np.float32)
    np.maximum.at(max_rho_per_bin, phi_indices, rho_clipped)
    # distance bool matrix generation
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
    def __init__(self, opt, seq, mode):
        self.opt = opt
        self.tile_size = self.opt['tiling']['tile_margin']
        self.seq =seq
        self.data_list, self.tile_manager = self.make_dataset(mode)
        self.mode = mode
        with open('conf/semantic-kitti.yaml', 'r') as f:
            self.semantic_kitti = yaml.safe_load(f)
        self.make_dataset()
        

    def make_dataset(self, mode = 'val'):
        if mode == 'train':
            sequence_list = ["01","02", "04", "05", "06", "08", "09", "10"]
            # sequence_list = ['00']
        elif mode == 'val':
            sequence_list = [self.seq]
        elif mode == 'test':
            sequence_list = ["00"]
        dataset = []
        tile_manager = {}
        for seq in sequence_list:
            # modifed 0324 for aligned test
            osm_gps_file_path = f'pose_data/kitti{seq}_0309_latlon_osm_pose.csv'
            osm_gps_file = np.array(pd.read_csv(osm_gps_file_path))
            if not os.path.exists(os.path.join(f"{self.opt['tiling']['tiles_path']}",f"100_tiles_test_{seq}.pkl")):
                    self.save_seq_tile_manager(seq, osm_gps_file)
            seq_tile_manager = self.load_seq_tile_manager(seq)
        
            # add data to data_list
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
        # print(seq_latlon.shape)
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
    
        # For OSM Tile
        seq_tile_manager = self.tile_manager[seq]
        latlon = np.array([lat, lon]) # [lat, lon] corresponds to the origin of point cloud in each frame. Here we transform it to the tile coordinate system
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
    pc_dataset = PcDataset_Val(opt, seq='00', mode=mode)
    pc_dataset[0]