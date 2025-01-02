import os
import numpy as np
import pandas as pd
from scipy.spatial import distance

# Parameters
num_bin = 360
N = 200  # Top N

# Paths (update paths as per your dataset structure)
base_data_dir = '/data/Pcmaploc/data/hand_crafted_data'
pc_descriptor_rot_inv_path = os.path.join(base_data_dir, 'kitti00_rotinv_lidar_descriptor.csv')
osm_descriptor_rot_inv_path = os.path.join(base_data_dir, 'kitti00_rotinv_osm_descriptor.csv')
pc_descriptor_path = os.path.join(base_data_dir, 'kitti00_lidar_descriptor.csv')
osm_descriptor_path = os.path.join(base_data_dir, 'kitti00_osm_descriptor.csv')
osm_pose_path = os.path.join(base_data_dir, 'kitti00_osm_pose.csv')
pc_pose_path = '/data/KITTI/2011_09_30/2011_09_30_drive_0027_sync/oxts/data'

# Load pose data
pc_files = sorted([f for f in os.listdir(pc_pose_path) if f.endswith('.txt')])
pc_poses = []
for filename in pc_files:
    current_path = os.path.join(pc_pose_path, filename)
    raw_data = np.loadtxt(current_path)
    pc_poses.append((raw_data[0], raw_data[1]))  # Assuming lat, lon are the first two columns
pc_poses = np.array(pc_poses)

osm_poses = pd.read_csv(osm_pose_path, header=None).values
pc_descriptor_rot_inv = pd.read_csv(pc_descriptor_rot_inv_path, header=None).values
osm_descriptor_rot_inv = pd.read_csv(osm_descriptor_rot_inv_path, header=None).values
pc_descriptor_list = pd.read_csv(pc_descriptor_path, header=None).values
osm_descriptor_list = pd.read_csv(osm_descriptor_path, header=None).values

# Rotation-invariant descriptor matching
topN_list = np.zeros((len(pc_descriptor_rot_inv), N), dtype=int)
matched_pair = np.zeros((len(pc_descriptor_rot_inv), 3))

for i, pc_descriptor in enumerate(pc_descriptor_rot_inv):
    if i % 100 == 0:
        print(f'Processing {i}/{len(pc_descriptor_rot_inv)}')

    diff_list = np.sum(np.abs(pc_descriptor - osm_descriptor_rot_inv), axis=1)
    match_idx = np.argmin(diff_list)
    matched_pair[i] = [i, match_idx, diff_list[match_idx]]

    topN_indices = np.argsort(diff_list)[:N]
    topN_list[i] = topN_indices

# Accuracy Calculation
result_N = np.zeros((len(matched_pair), N))
for i, (pc_idx, match, _) in enumerate(matched_pair):
    pc_x, pc_y = pc_poses[int(pc_idx)]
    topN_distances = [
        distance.euclidean((osm_poses[idx, 1], osm_poses[idx, 0]), (pc_x, pc_y))
        for idx in topN_list[i]
    ]
    result_N[i] = [d < 5 for d in topN_distances]

top_N_accuracy = np.array([np.mean(np.any(result_N[:, :n], axis=1)) for n in range(1, N + 1)])
print(f'Stage 1 Top 1 Accuracy: {top_N_accuracy[0] * 100:.2f}%')
print(f'Stage 1 Top 5 Accuracy: {top_N_accuracy[4] * 100:.2f}%')
print(f'Stage 1 Top 10 Accuracy: {top_N_accuracy[9] * 100:.2f}%')

# Full descriptor comparison
matched_pair_full = np.zeros((len(pc_descriptor_list), 3))
topN_list_full = np.zeros((len(pc_descriptor_list), N), dtype=int)

for i, pc_descriptor in enumerate(pc_descriptor_list):
    if i % 100 == 0:
        print(f'Processing full descriptor {i}/{len(pc_descriptor_list)}')

    diff_list = []
    for j in topN_list[i]:
        osm_descriptor = osm_descriptor_list[j]
        min_diff_k = np.inf
        for k in range(num_bin):
            osm_shifted = np.roll(osm_descriptor, k)
            diff_k = np.sum(np.abs(pc_descriptor - osm_shifted))
            min_diff_k = min(min_diff_k, diff_k)
        diff_list.append(min_diff_k)

    match_idx = np.argmin(diff_list)
    matched_pair_full[i] = [i, topN_list[i][match_idx], diff_list[match_idx]]
    topN_list_full[i] = topN_list[i][np.argsort(diff_list)]

result_N_full = np.zeros((len(matched_pair_full), N))
for i, (pc_idx, match, _) in enumerate(matched_pair_full):
    pc_x, pc_y = pc_poses[int(pc_idx)]
    topN_distances = [
        distance.euclidean((osm_poses[idx, 1], osm_poses[idx, 0]), (pc_x, pc_y))
        for idx in topN_list_full[i]
    ]
    result_N_full[i] = [d < 5 for d in topN_distances]

top_N_accuracy_full = np.array([np.mean(np.any(result_N_full[:, :n], axis=1)) for n in range(1, N + 1)])
print(f'Stage 2 Top 1 Accuracy: {top_N_accuracy_full[0] * 100:.2f}%')
print(f'Stage 2 Top 5 Accuracy: {top_N_accuracy_full[4] * 100:.2f}%')
print(f'Stage 2 Top 10 Accuracy: {top_N_accuracy_full[9] * 100:.2f}%')
