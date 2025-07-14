#!/bin/bash

# Create target directory
mkdir -p ~/kitti/odometry
cd ~/kitti/odometry

# Download required datasets
#echo "Downloading color images (65 GB)..."
#wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip

#echo "Downloading calibration files (1 MB)..."
#wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip

# echo "Downloading Velodyne laser scans (80 GB)..."
# wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip

echo "Downloading ground truth poses (4 MB)..."
 wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip

# Unzip everything
echo "Extracting files..."
#unzip -n data_odometry_color.zip
#unzip -n data_odometry_calib.zip
#unzip -n data_odometry_velodyne.zip
unzip -n data_odometry_poses.zip

echo "✅ KITTI Odometry dataset is ready!"
