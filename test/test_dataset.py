import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from kitti_dataloader import PcMapLocDataset
from maploc.utils.geo import Projection, BoundaryBox
import yaml
from maploc.osm.viz import GeoPlotter

def plot_lat_lon_on_osm(dataset, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    seq_list = ['00', '01', '02', '04', '05', '06', '07', '08', '09', '10']
    for seq in seq_list:
        lat_lon_list = []
        for data_item in dataset.data_list:
            if data_item['seq'] == seq:
                # print(data_item)
                lat = data_item['lat']
                lon = data_item['lon']
                lat_lon_list.append((lat, lon))
        print(f"Sequence {seq} has {len(lat_lon_list)} items") 
        if not lat_lon_list:
            continue

        lat_lon_array = np.array(lat_lon_list)
        projection = dataset.tile_manager[seq].projection
        xy_coords = projection.project(lat_lon_array)
        
        # 获取 OSM 底图
        tile_margin = dataset.tile_manager[seq].tile_size
        bbox_map_min = np.floor(xy_coords.min(0) / tile_margin) * tile_margin 
        bbox_map_max = np.ceil(xy_coords.max(0) / tile_margin) * tile_margin
        sample_bbox = BoundaryBox(bbox_map_min, bbox_map_max)+tile_margin
        # sample_bbox = BoundaryBox(xy_coords.min(axis=0), xy_coords.max(axis=0))
        canvas = dataset.tile_manager[seq].query(sample_bbox)
        osm_map = canvas.raster
        # print(xy_coords[0])
        
        xy_coords = canvas.to_uv(xy_coords)
        fig, ax = plt.subplots()
        ax.set_title(f'Sequence {seq}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # 假设 osm_map 是一个 NumPy 数组
        ax.imshow(osm_map[0])

        # 绘制 xy 坐标
        ax.scatter(xy_coords[:, 0], xy_coords[:, 1], c='red', s=10, label='Lat/Lon Points')
        ax.legend()

        # 保存图像
        output_path = os.path.join(output_dir, f'pics/dataset_sequence_{seq}.png')
        plt.savefig(output_path)
        # plt.show()
        plt.close(fig)
        # break
        

# 示例用法
if __name__ == "__main__":
    with open("conf/data/kitti.yaml", "r") as file:
        opt = yaml.safe_load(file)
    data_dir = opt['loading']['pc_data_dir']
    mode = 'val'
    dataset = PcMapLocDataset(opt, mode=mode)
    # dataset = PcMapLocDataset('path/to/dataset')
    output_dir = './'
    plot_lat_lon_on_osm(dataset, output_dir)