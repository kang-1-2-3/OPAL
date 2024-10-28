# 11. Geo-localization with point clouds and OpenStreetMap

## Project status
### 1 Paper Review 
[Google Sheet Link](https://docs.google.com/spreadsheets/d/1CB7xgJA1dnAZfQlWCjkckuNveE-z1RAy5DIpht9DmxU/edit?usp=sharing)

### 2 Dataset Preparation 
- [ ] Nuscenes and Argoverse 2 dataset download
- [ ] Corresponding OSM data download
- [ ] Dataset Map and OSM alignment check

### 3 Network Design
#### 3.1 Visual Place Recognition (VPR)

### 4 Current Unsolved Problems 
- The alignment between OSM and Nuscenes map is not very well, obvious drift can be seen in the figures.

![Boston Seaport](boston-seaport.png)
Boston Seaport 

![Singapore Holland Village](singapore-hollandvillage.png)
Singapore Holland Village

![Singapore One North](singapore-onenorth.png)
Singapore One North

![Singapore Queenstown](singapore-queenstown.png)
Singapore Queenstown

- The storage and GPU in lab 1778 is not sufficient for this task.