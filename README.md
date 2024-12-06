# 11. Geo-localization with point clouds and OpenStreetMap

## Project status
### 1 Paper Review 
[Google Sheet Link](https://docs.google.com/spreadsheets/d/1CB7xgJA1dnAZfQlWCjkckuNveE-z1RAy5DIpht9DmxU/edit?usp=sharing)

### 2 Dataset Preparation 
- [X] Nuscenes and Argoverse 2 dataset download
- [X] Corresponding OSM data download
- [X] Dataset Map and OSM alignment check

### 3 Network Design

#### 3.1 Point Cloud BEV Representation 
Options: 
- [X] BEV Image Generation [(ref)](https://arxiv.org/pdf/2408.01841)
- [ ] PillarNext [(ref)](https://github.com/qcraftai/pillarnext?tab=readme-ov-file)

#### 3.2 OSM Representation 
Options:
- [X] Raster Image [(ref)](https://github.com/facebookresearch/OrienterNet/tree/main)
- [ ] Proposed OSM representation

#### 3.3 Visual Place Recognition (VPR)
The OSM is divided into tiles for each sequence with the size of 64m*64m. The point cloud is transformed to BEV image with pixel size of 0.4m, leading to (201, 201) for each frame of point cloud. Here we use [BEVPlace++](https://arxiv.org/pdf/2408.01841) for point cloud global descriptor generation and [OrienterNet](https://github.com/facebookresearch/OrienterNet/tree/main) for OSM tile feature extraction. There are two options for OSM tile global feature extraction:
- [X] Average pooling with fully connected layers
- [ ] NetVLAD
 
### 4 Current Unsolved Problems 
[Current problems.md](Current_problems.md)

### 5 Progress
[Progress.md](./Progress.md)