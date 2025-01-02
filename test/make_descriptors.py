import os
import numpy as np
import json
import utm
import csv
from scipy.spatial import distance
from scipy.interpolate import interp1d
from tqdm import tqdm

# {
# Nr.     Sequence name     Start   End
# ---------------------------------------
# 00: 2011_10_03_drive_0027 000000 004540
# 02: 2011_10_03_drive_0034 000000 004660
# 05: 2011_09_30_drive_0018 000000 002760
# 06: 2011_09_30_drive_0020 000000 001100
# 07: 2011_09_30_drive_0027 000000 001100
# 08: 2011_09_30_drive_0028 001100 005170
# 09: 2011_09_30_drive_0033 000000 001590
# 10: 2011_09_30_drive_0034 000000 001200

# Same Places:
# (00 & 07), (05 & 06), (09 & 10)
# }

seq_names = ['00']

osm = True
lidar = False

num_bin = 360
sensor_range = 50
bin_size = 5  # bin size of rotation-invariant descriptor

def interpolate_road_coords(x_coords, y_coords):
    """
    使用与MATLAB相似的方式插值道路坐标
    """
    interp_x = []
    interp_y = []
    
    for i in range(len(x_coords) - 1):
        x1, x2 = x_coords[i], x_coords[i + 1]
        y1, y2 = y_coords[i], y_coords[i + 1]
        dx = x1 - x2
        dy = y1 - y2
        
        # 计算插值步长
        step = abs(dx / np.sqrt(dx**2 + dy**2))
        itp_x = np.arange(min(x1, x2), max(x1, x2), step)
        itp_x = np.append(itp_x, max(x1, x2))  # 确保包含最大值
        
        # 确保插值点在范围内
        itp_x = itp_x[itp_x <= max(x1, x2)]
        itp_x = itp_x[itp_x >= min(x1, x2)]
        
        # 确保插值范围不重复
        if len(interp_x) > 0 and itp_x[0] == interp_x[-1]:
            itp_x = itp_x[1:]
        
        # 使用线性插值计算对应的 y 值
        f = interp1d([x1, x2], [y1, y2], bounds_error=False, fill_value="extrapolate")
        itp_y = f(itp_x)
        
        # 合并当前段插值结果
        interp_x.extend(itp_x)
        interp_y.extend(itp_y)
    
    return np.array(interp_x), np.array(interp_y)

def deg2utm(lat, lon):
    import numpy as np

    sa = 6378137.000000
    sb = 6356752.314245
    e2 = np.sqrt((sa ** 2 - sb ** 2)) / sb
    e2cuadrada = e2 ** 2
    c = (sa ** 2) / sb

    lat = np.radians(lat)
    lon = np.radians(lon)
    Huso = np.floor((lon / np.radians(6)) + 31)
    S = (Huso * 6) - 183
    deltaS = lon - np.radians(S)

    a = np.cos(lat) * np.sin(deltaS)
    epsilon = 0.5 * np.log((1 + a) / (1 - a))
    nu = np.arctan(np.tan(lat) / np.cos(deltaS)) - lat
    v = (c / np.sqrt(1 + (e2cuadrada * (np.cos(lat) ** 2)))) * 0.9996
    ta = (e2cuadrada / 2) * epsilon ** 2 * (np.cos(lat) ** 2)
    a1 = np.sin(2 * lat)
    a2 = a1 * (np.cos(lat) ** 2)
    j2 = lat + (a1 / 2)
    j4 = ((3 * j2) + a2) / 4
    j6 = ((5 * j4) + (a2 * (np.cos(lat) ** 2))) / 3
    alfa = (3 / 4) * e2cuadrada
    beta = (5 / 3) * alfa ** 2
    gama = (35 / 27) * alfa ** 3
    Bm = 0.9996 * c * (lat - alfa * j2 + beta * j4 - gama * j6)
    xx = epsilon * v * (1 + (ta / 3)) + 500000
    yy = nu * v * (1 + ta) + Bm
    yy[yy < 0] += 9999999

    return xx, yy

for seq_name in seq_names:
    print(f'KITTI sequence {seq_name} start.')

    # Load OSM files
    building_file_name = f'/data/Pcmaploc/data/hand_crafted_data/kitti{seq_name}_buildings.geojson'
    road_file_name = f'/data/Pcmaploc/data/hand_crafted_data/kitti{seq_name}_roads.geojson'
    osm_save_path = f'/data/Pcmaploc/data/hand_crafted_data/kitti{seq_name}_osm_descriptor.npy'
    osm_rotinv_save_path = f'/data/Pcmaploc/data/hand_crafted_data/kitti{seq_name}_rotinv_osm_descriptor.npy'
    lidar_save_path = f'/data/Pcmaploc/data/hand_crafted_data/kitti{seq_name}_lidar_descriptor.npy'
    lidar_rotinv_save_path = f'/data/Pcmaploc/data/hand_crafted_data/kitti{seq_name}_rotinv_lidar_descriptor.npy'

    with open(building_file_name, 'r') as f:
        building_val = json.load(f)
    
    with open(road_file_name, 'r') as f:
        road_val = json.load(f)

    # 投影所有的building的坐标
    x_coords_out = []
    y_coords_out = []
    for feature in building_val['features']:
        geometry = feature['geometry']
        coordinates = geometry['coordinates']
        if isinstance(coordinates, list):
            coords_out = np.array(coordinates[0][0]).reshape(-1, 2)
        else:
            coords_out = np.array(coordinates).reshape(-1, 2)

        x_coords_out_temp, y_coords_out_temp = deg2utm(coords_out[:, 1], coords_out[:, 0])
        x_coords_out.append(x_coords_out_temp)
        y_coords_out.append(y_coords_out_temp)

    if osm:
        coords_x, coords_y = [], []
        for feature in road_val['features']:
            geometry = feature['geometry']
            coordinates = geometry['coordinates']
            if isinstance(coordinates, list):
                coords_out = np.array(coordinates)
            else:
                coords_out = np.array(coordinates).reshape(-1, 2)

            x_coords, y_coords = deg2utm(coords_out[:, 1], coords_out[:, 0])

            interp_x, interp_y = interpolate_road_coords(x_coords, y_coords)
            coords_x.extend(interp_x)
            coords_y.extend(interp_y)

        coords = np.column_stack((coords_x, coords_y))
        osm_pose_path = f'/data/Pcmaploc/data/hand_crafted_data/kitti{seq_name}_osm_pose.npy'
        np.save(osm_pose_path, coords)

        descriptor = np.zeros((len(coords_x), num_bin))
        for i, (x, y) in tqdm(enumerate(zip(coords_x, coords_y)), total=len(coords_x), desc="Generating OSM descriptor"):
            descriptor_i = np.zeros(num_bin)
            for j in range(num_bin):
                target_angle = j * 2 * np.pi / num_bin
                shortest_d = sensor_range

                for k in range(len(x_coords_out)):
                    x_coords_out_temp = x_coords_out[k]
                    y_coords_out_temp = y_coords_out[k]

                    distances = (x_coords_out_temp - x) ** 2 + (y_coords_out_temp - y) ** 2
                    if min(distances) > sensor_range ** 2:
                        continue

                    for l in range(len(x_coords_out_temp)):
                        vertex1 = [x_coords_out_temp[l % len(x_coords_out_temp)], y_coords_out_temp[l % len(y_coords_out_temp)]]
                        vertex2 = [x_coords_out_temp[(l + 1) % len(x_coords_out_temp)], y_coords_out_temp[(l + 1) % len(y_coords_out_temp)]]

                        theta1 = np.arctan2(vertex1[1] - y, vertex1[0] - x)
                        theta2 = np.arctan2(vertex2[1] - y, vertex2[0] - x)

                        if theta1 < 0:
                            theta1 += 2 * np.pi
                        if theta2 < 0:
                            theta2 += 2 * np.pi

                        max_theta = max(theta1, theta2)
                        theta1 += np.pi - max_theta
                        theta2 += np.pi - max_theta
                        target_angle += np.pi - max_theta

                        if theta1 < 0:
                            theta1 += 2 * np.pi
                        elif theta1 > 2 * np.pi:
                            theta1 -= 2 * np.pi
                        if theta2 < 0:
                            theta2 += 2 * np.pi
                        elif theta2 > 2 * np.pi:
                            theta2 -= 2 * np.pi
                        if target_angle < 0:
                            target_angle += 2 * np.pi
                        elif target_angle > 2 * np.pi:
                            target_angle -= 2 * np.pi

                        if np.median([target_angle, theta1, theta2]) == target_angle:
                            if min([target_angle, theta1, theta2]) == theta1:
                                a = np.linalg.norm([x - vertex1[0], y - vertex1[1]])
                                b = np.linalg.norm([x - vertex2[0], y - vertex2[1]])
                                phi1 = target_angle - theta1
                                phi2 = theta2 - target_angle
                            else:
                                a = np.linalg.norm([x - vertex2[0], y - vertex2[1]])
                                b = np.linalg.norm([x - vertex1[0], y - vertex1[1]])
                                phi1 = target_angle - theta2
                                phi2 = theta1 - target_angle

                            d = (a * b * np.sin(phi1 + phi2)) / (a * np.sin(phi1) + b * np.sin(phi2))
                            shortest_d = min(d, shortest_d)

                descriptor_i[j] = 0 if shortest_d == sensor_range else shortest_d
            descriptor[i] = descriptor_i

        np.save(osm_save_path, descriptor)

        rot_inv_descriptor = np.zeros((len(descriptor), sensor_range // bin_size))
        for i, row in tqdm(enumerate(descriptor), total=len(descriptor), desc="Generating rotation-invariant OSM descriptor"):
            hist = np.histogram(row, bins=np.arange(0, sensor_range + bin_size, bin_size))[0]
            rot_inv_descriptor[i] = hist

        np.save(osm_rotinv_save_path, rot_inv_descriptor)

        print(f'KITTI sequence {seq_name} OSM descriptor is generated.')
    else:
        print('Skip OSM descriptor.')

    # make LiDAR descriptor
    if lidar:
        pc_path = f'/data/Pcmaploc/data/kitti/building_pointclouds/{seq_name}'  # Building pointcloud path (folder)
        files = os.listdir(pc_path)
        files = [f for f in files if f.endswith('.npy')]
        descriptor_list = np.zeros((len(files), num_bin))
        for i, filename in tqdm(enumerate(files), total=len(files), desc="Generating LiDAR descriptor"):
            current_path = os.path.join(pc_path, filename)
            pc = np.load(current_path)

            tan_list = np.arctan2(pc[:, 1], pc[:, 0])
            tan_list = tan_list + (tan_list < 0) * 2 * np.pi

            descriptor = np.zeros(num_bin)
            for j in range(num_bin):
                target_angle_orig = j * 2 * np.pi / num_bin

                target_angle = np.pi
                tan_list_temp = tan_list + np.pi - target_angle_orig
                tan_list_temp = tan_list_temp + (tan_list_temp < 0) * 2 * np.pi
                tan_list_temp = tan_list_temp - (tan_list_temp > 2 * np.pi) * 2 * np.pi
                target_points = pc[(tan_list_temp > target_angle - np.pi / num_bin) & (tan_list_temp < target_angle + np.pi / num_bin), :]
                if len(target_points) == 0:
                    descriptor[j] = 0
                else:
                    if np.min(np.sqrt(target_points[:, 0] ** 2 + target_points[:, 1] ** 2)) > sensor_range:
                        descriptor[j] = 0
                    else:
                        descriptor[j] = np.min(np.sqrt(target_points[:, 0] ** 2 + target_points[:, 1] ** 2))
            descriptor_list[i, :] = descriptor

        np.save(lidar_save_path, descriptor_list)

        pc_descriptor_rot_inv = np.zeros((len(descriptor_list), sensor_range // bin_size))
        for i, row in tqdm(enumerate(descriptor_list), total=len(descriptor_list), desc="Generating rotation-invariant LiDAR descriptor"):
            range_angle_descriptor = np.zeros((num_bin, sensor_range // bin_size))

            pc_descriptor = row
            for j in range(num_bin):
                if pc_descriptor[j] != 0:
                    range_angle_descriptor[j, int(np.ceil(pc_descriptor[j] / bin_size)) - 1] = 1
            rot_inv_descriptor = np.sum(range_angle_descriptor, axis=0)

            pc_descriptor_rot_inv[i, :] = rot_inv_descriptor

        np.save(lidar_rotinv_save_path, pc_descriptor_rot_inv)

        print(f'KITTI sequence {seq_name} LiDAR descriptor is generated.')
    else:
        print('Skip LiDAR descriptor.')
