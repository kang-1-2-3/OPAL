import torch.utils.data as data
import numpy as np 
import pykitti 
import os 
import yaml
from maploc.utils.geo import Projection
from pathlib import Path
from maploc.osm.tiling import TileManager, BoundaryBox
import torch
import numba as nb


class PcMapLocDataset(data.Dataset):
    def __init__(self, opt, mode):
        self.opt = opt
        self.tile_size = self.opt['tiling']['tile_margin']
        self.data_list, self.tile_manager = self.make_dataset(mode)
        self.mode = mode
        self.grid_size = np.asarray([480,360,32])
        self.ignore_label = 255
        with open('conf/semantic-kitti.yaml', 'r') as f:
            self.semantic_kitti = yaml.safe_load(f)

    def make_dataset(self, mode = 'train'):
        if mode == 'train':
            sequence_list = self.opt['loading']['dataset_split']['train']
        elif mode == 'val':
            sequence_list = self.opt['loading']['dataset_split']['val']

        dataset = []
        tile_manager = {}
        for seq in sequence_list:
            pc_gps_file_path = os.path.join(f"{self.opt['loading']['gps_data_dir']}", f"gps_sequence_{seq}.npy")
            assert os.path.exists(pc_gps_file_path)
            pc_gps_file = np.load(pc_gps_file_path, allow_pickle=True).item()
            kitti_dataset = pykitti.odometry(self.opt['loading']['pc_data_dir'], seq)
            if not os.path.exists(os.path.join(f"{self.opt['tiling']['tiles_path']}",f"100_tiles_{seq}.pkl")):
                self.save_seq_tile_manager(seq, pc_gps_file)
            seq_tile_manager = self.load_seq_tile_manager(seq)
            # add data to data_list
            for index, (pc_file_path, T_w_velo_i, lat, lon) in enumerate(zip(kitti_dataset.velo_files, pc_gps_file['T_w_velo_i'], pc_gps_file['lat'], pc_gps_file['lon'])):
                dataset.append({'seq': seq, 
                                'index': index, 
                                'pc_file_path': pc_file_path, 
                                'gps_trans_data': T_w_velo_i,
                                'lat': lat, 
                                'lon': lon})
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
        seq_tile_manager.save(os.path.join(self.opt['tiling']['tiles_path'], f"{seq}.pkl"))

    def load_seq_tile_manager(self, seq):
        tile_manager = TileManager.load(Path(os.path.join(self.opt['tiling']['tiles_path'], f"100_tiles_{seq}.pkl")))
        return tile_manager

    def cart2polar(self, input_xyz):
        rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
        phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
        return np.stack((rho,phi,input_xyz[:,2]),axis=1)
    
    def polar2cat(self, input_xyz_polar):
        x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
        y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
        return np.stack((x,y,input_xyz_polar[2]),axis=0)

    def random_rot90(self, raster, xy=None, heading=None, seed=None):
        rot = np.random.RandomState(seed).randint(0, 4)
        raster = np.rot90(raster, rot, axes=(-2, -1))
        return raster, rot

    def random_flip(self, raster, xy=None, seed=None):
        state = np.random.RandomState(seed)
        flip_x = state.rand() > 0.5  # flip flag
        flip_y = False
        if not flip_x:
            flip_y = state.rand() > 0.5
        
        if flip_x:
            raster = raster[..., :, ::-1]
        elif flip_y:
            raster = raster[..., ::-1, :]
        return raster, (flip_x, flip_y)

    
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
    
    def __len__(self):
        return len(self.data_list)
    
    def getpcaug(self, index,theta):
        data_item = self.data_list[index]
        pc_file_path = data_item['pc_file_path']

        pcs = np.fromfile(pc_file_path,dtype=np.float32).reshape(-1,4)
        pcs = self.augment_point_cloud_with_2d_rotation(pcs,theta)

        label_path = pc_file_path.replace('/velodyne/', '/predictions/').replace('.bin', '.label') # rangenet
        raw_label = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF

        inline_mask = (np.sqrt(pcs[:, 0]**2 + pcs[:, 1]**2) < 50) & (np.abs(pcs[:, 2]) < 3)
        pcs = pcs[inline_mask]
        raw_label = raw_label[inline_mask]
        

        learning_map = self.semantic_kitti['learning_map']
        map_func = np.vectorize(lambda x: learning_map.get(x))  
        labels = map_func(raw_label).reshape((-1,1))

        

        xyz_pol = self.cart2polar(pcs) # [N, 3] rho, phi, z
        max_bound = np.asarray([50,np.pi,1.5])
        min_bound = np.asarray([3,-np.pi,-3])

        # get grid index
        crop_range = max_bound - min_bound 
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1)

        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int32) # [N, 3]

        distance_feature_2d_original = compute_distance_feature_polar(xyz_pol)
        
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,pcs[:,:2]),axis = 1)

        return_fea = return_xyz
    
        data_tuple = ([torch.from_numpy(return_fea).float()],[torch.from_numpy(grid_ind)],[torch.from_numpy(labels)], torch.from_numpy(distance_feature_2d_original)[None,...].float())
        
        return data_tuple

    def __getitem__(self, index):
        data_item = self.data_list[index]
        seq, seq_index, pc_file_path, gps_trans_data, lat, lon = data_item['seq'], data_item['index'], data_item['pc_file_path'], data_item['gps_trans_data'], data_item['lat'], data_item['lon']
        
        # For OSM Tile
        seq_tile_manager = self.tile_manager[seq]
        latlon = np.array([lat, lon]) # [lat, lon] corresponds to the origin of point cloud in each frame. Here we transform it to the tile coordinate system
        proj = seq_tile_manager.projection
        xy = proj.project(latlon)
        sample_bbox = BoundaryBox(xy-self.tile_size//2, xy+self.tile_size//2)
        canvas = seq_tile_manager.query(sample_bbox)
        raster = canvas.raster

        pcs = np.fromfile(pc_file_path,dtype=np.float32).reshape(-1,4)
        
        # Add data augmentation for OSM
        if self.mode == "train":
            # Apply augmentation to OSM and record parameters
            raster, rot_k = self.random_rot90(raster)
            raster, (flip_x, flip_y) = self.random_flip(raster)
            
            # Add grid mask augmentation
            # raster, grid_mask_pattern = self.grid_mask(raster, mask_ratio=0.3, grid_size=32)
        
        if self.mode == 'train':
            pcs = self.augment_point_cloud_with_2d_rotation(pcs)
        label_path = pc_file_path.replace('/velodyne/', '/pred_labels/').replace('.bin', '.label')
        # label_path = pc_file_path.replace('/velodyne/', '/labels/').replace('.bin', '.label')
        # label_path = pc_file_path.replace('/velodyne/', '/predictions/').replace('.bin', '.label') # rangenet
        raw_label = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF

        inline_mask = (np.sqrt(pcs[:, 0]**2 + pcs[:, 1]**2) < 50) & (np.abs(pcs[:, 2]) < 3)
        intensity = pcs[:, 3][inline_mask] # add intensity
        pcs = pcs[inline_mask]
        raw_label = raw_label[inline_mask]
        
        learning_map = self.semantic_kitti['learning_map']
        map_func = np.vectorize(lambda x: learning_map.get(x))  
        labels = map_func(raw_label).reshape((-1,1))

        xyz_pol = self.cart2polar(pcs) # [N, 3] rho, phi, z
        max_bound = np.asarray([50,np.pi,1.5])
        min_bound = np.asarray([3,-np.pi,-3])

        # get grid index
        crop_range = max_bound - min_bound # [47, 2pi, 4.5]
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1) # [0.09812108559498957, 0.01750190893364787, 0.14516129032258066]

        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int32) # [N, 3]

        distance_feature_2d_original = compute_distance_feature_polar(xyz_pol)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,pcs[:,:2]),axis = 1)

        return_fea = return_xyz
        
        if self.mode != 'train':
            data_tuple = (grid_ind,labels,return_fea, intensity)
        else:
            data_tuple = (grid_ind,labels,return_fea, intensity)
        return {'data_tuple': data_tuple,
            'osm_map': torch.from_numpy(np.ascontiguousarray(raster)).long(),
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
    intensity = [d['data_tuple'][3] for d in data]
    osm_map = torch.from_numpy(np.stack([d['osm_map'] for d in data]))
    xy = torch.from_numpy(np.stack([d['xy'] for d in data]))
    pc_vis_mask = torch.from_numpy(np.stack([d['pc_vis_mask'] for d in data]))
    return grid_ind_stack,point_label,xyz, osm_map, xy, pc_vis_mask, intensity

def collate_fn_BEV_test(data):    
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),grid_ind_stack,point_label,xyz,index

def compute_distance_feature_polar(xyz_pol, r_bins=480, phi_bins=360, r_min=3, r_max=50):
    """
    Compute distance feature using NumPy vectorization directly based on polar coordinates xyz_pol.
    xyz_pol: [N, 3], (rho, phi, z).
    """
    phi_step = (2*np.pi) / phi_bins
    rspace = np.linspace(r_min, r_max, r_bins)
    # Filter points outside radius range
    mask = (xyz_pol[:, 0] >= r_min) & (xyz_pol[:, 0] <= r_max)
    rho_clipped = xyz_pol[mask, 0]
    phi_clipped = xyz_pol[mask, 1]
    # Calculate phi index for each point
    phi_indices = ((phi_clipped + np.pi) / phi_step).astype(int)
    phi_indices = np.clip(phi_indices, 0, phi_bins - 1)
    # Aggregate maximum rho for each phi bin
    max_rho_per_bin = np.zeros(phi_bins, dtype=np.float32)
    np.maximum.at(max_rho_per_bin, phi_indices, rho_clipped)
    # Generate boolean matrix for distance feature
    distance_feature = (rspace[None] <= max_rho_per_bin[:, None])
    return distance_feature.T


def compute_distance_feature_polar_hist(xyz_pol, phi_bins=360, r_min=3, r_max=50):
    """
    Compute maximum distance (distance feature) for each angle.
    
    xyz_pol: [N, 3], (rho, phi, z).
    Returns: (360,) array, maximum rho value for each angle.
    """
        
    phi_step = (2 * np.pi) / phi_bins

    # Filter points outside radius range
    mask = (xyz_pol[:, 0] >= r_min) & (xyz_pol[:, 0] <= r_max)
    rho = xyz_pol[mask, 0]
    phi = xyz_pol[mask, 1]

    # Calculate phi index for each point
    phi_indices = ((phi + np.pi) / phi_step).astype(int)
    phi_indices = np.clip(phi_indices, 0, phi_bins - 1)

    # Calculate maximum rho within each phi bin
    max_rho_per_bin = np.zeros(phi_bins, dtype=np.float32)
    np.maximum.at(max_rho_per_bin, phi_indices, rho)

    return max_rho_per_bin

if __name__ == "__main__":
    with open("conf/data/kitti.yaml", "r") as file:
        opt = yaml.safe_load(file)
    data_dir = opt['loading']['pc_data_dir']
    mode = 'train'
    
    dataset = PcMapLocDataset(opt, mode=mode)
    dataset[0]