#!/bin/bash

output_dir="/home/data_sata/Pcmaploc/data/raw_kitti"
mkdir -p $output_dir

# 配置代理（HTTP 和 HTTPS）
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

files=(
2011_09_30_calib.zip
2011_09_30_drive_0018
2011_09_30_drive_0020
2011_09_30_drive_0027
2011_09_30_drive_0028
2011_09_30_drive_0033
2011_09_30_drive_0034
2011_10_03_calib.zip
2011_10_03_drive_0027
2011_10_03_drive_0034)

for i in ${files[@]}; do
    if [ ${i:(-3)} != "zip" ]; then
        shortname=$i'_sync.zip'
        fullname=$i'/'$i'_sync.zip'
    else
        shortname=$i
        fullname=$i
    fi

    echo "Downloading: $shortname"
    
    # 下载文件
    if wget --show-progress --no-check-certificate 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname; then
        echo "Download successful: $shortname"
        
        # 解压文件
        if unzip -o $shortname -d $output_dir; then
            rm $shortname
        else
            echo "Failed to unzip $shortname"
        fi
    else
        echo "Failed to download $shortname"
    fi
done

