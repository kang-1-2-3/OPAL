#!/bin/bash

# 创建下载目录
DOWNLOAD_DIR="/home/data_sata/Pcmaploc/data/nus"
mkdir -p $DOWNLOAD_DIR

# 文件列表
URLS=(
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz"
  "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz"
  "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_blobs.tgz"
  "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz"
  "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz"
  "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval09_blobs.tgz"
  "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval10_blobs.tgz"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-test_meta.tgz"
  "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs.tgz"
)

# 并行下载函数
download_file() {
  local url=$1
  local dir=$2
  wget -c --progress=bar:force:noscroll -P "$dir" "$url"
}

export -f download_file

# 使用 xargs 并行处理下载
echo "${URLS[@]}" | tr ' ' '\n' | xargs -n 1 -P 1 -I {} bash -c 'download_file "$@"' _ {} $DOWNLOAD_DIR

echo "All files downloaded to $DOWNLOAD_DIR."
