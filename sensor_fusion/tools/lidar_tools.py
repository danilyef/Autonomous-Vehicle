from easydict import EasyDict as edict
import numpy as np
from PIL import Image
from numpy.lib.function_base import percentile
from sfa.data_process.kitti_bev_utils import get_corners


def create_lidar_config():
    lidar_config = edict()
    lidar_config.points_per_layer = 3072
    lidar_config.n_layers = 64
    lidar_config.fov_up = 25
    lidar_config.fov_down = -5

    return lidar_config


def pointcloud_to_range_image(pcl:np.ndarray, lidar_config: edict) -> np.ndarray:
    width = lidar_config.points_per_layer
    height = lidar_config.n_layers
    fov_up = lidar_config.fov_up * np.pi /180
    fov_down = lidar_config.fov_down * np.pi / 180.0
    
    # Step 1:
    # Calculate range and angles for each point
    x = pcl[:, 0]
    y = pcl[:, 1]
    z = pcl[:, 2]
    channels_info = pcl[:, 3:]

    ranges = np.sqrt(x*x+y*y+z*z)
    azimuths = np.arctan2(y, x)
    elevations = np.arcsin(z / ranges)

    # Step 2:
    elevations[np.isnan(elevations)] = 0
    
    # Step 3:
    vs = np.round(((fov_up - elevations) / (fov_up - fov_down)) * (height-1)).astype(np.int32) 
    #Columns index of the image
    us = np.round(((azimuths + np.pi) / (2*np.pi)) * (width - 1)).astype(np.int32)
    
    # Create image placeholders filled with 0.
    range_image = np.zeros((height, width), dtype=np.float32)
    intensity_image = np.zeros((height, width, pcl.shape[1]-3), dtype=np.float32)
    # Populate image
    range_image[vs, us] = ranges
    intensity_image[vs, us] = channels_info
    
    #  Stack channels as we also want to add the intensity to our image
    range_channels_image = np.dstack((range_image, intensity_image))
    
    return range_channels_image

def show_range_image(range_image):
    ri = range_image.copy()
    deg45 = int(ri.shape[1]/8)
    ri_center = int(ri.shape[1]/2)
    ri = ri[:,ri_center-deg45:ri_center+deg45,:]
    #  map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    ri_range = ri[:,:,0]
    ri_range = (ri_range-np.amin(ri_range)) * 255 / (np.amax(ri_range) - np.amin(ri_range))
    # map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    ri_intensity = ri[:,:,1]
    p99 = percentile(ri_intensity,99)
    p1  = percentile(ri_intensity,1), 
    ri_intensity = np.clip(ri_intensity, p1, p99)
    ri_intensity = 255*(ri_intensity-p1)/(p99-p1)
    # stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer   
    img_range_intensity = np.vstack((ri_range, ri_intensity)).astype(np.uint8)
    Image.fromarray(ri_range.astype(np.uint8)).show()
    Image.fromarray(ri_intensity.astype(np.uint8)).show()
    return img_range_intensity


def pointcloud_from_range_image(ri, lidar_config):
    # Step 1: 
    # Convert FOV values from degrees to radians
    fov_up = lidar_config.fov_up * np.pi / 180.0
    fov_down = lidar_config.fov_down * np.pi / 180.0
    
    # Calculate vertical and horizontal FOV angles
    height, width, channels = ri.shape
    
    # Step 2: 
    # Calculate range and angles for each pixel
    elevations = np.linspace(fov_up, fov_down, height)
    azimuths = np.linspace(-np.pi, np.pi, width)
    azimuths, elevations = np.meshgrid(azimuths, elevations)
    ranges = ri[:,:,0]
    
    # Step 3: 
    # Convert range and angles to Cartesian coordinates
    x = ranges * np.cos(elevations) * np.cos(azimuths)
    y = ranges * np.cos(elevations) * np.sin(azimuths)
    z = ranges * np.sin(elevations)
    
    # Step 4:
    # Create empty point cloud array
    pcl = np.zeros((height * width, 3+(channels-1)), dtype=np.float32)
    # Fill in point cloud array
    pcl[:, 0] = x.flatten()
    pcl[:, 1] = y.flatten()
    pcl[:, 2] = z.flatten()

    pcl[:,3:] = np.reshape(ri[:,:,1:], (height*width,-1))
    
    # Step 5:
    # Filter out invalid points
    valid_indices = np.where(np.isfinite(pcl).all(axis=1))[0]
    pcl = pcl[valid_indices]
    
    return pcl



def label_extraction(frame,bev_configs):
    labels_detections = []
    for i in range(len(frame.lidars[0].detections)):
        detection = frame.lidars[0].detections[i]
        y, x, z = detection.pos
        l, w, h = detection.scale
        _, _, yaw = detection.rot

        x = (x - bev_configs.lim_x[0]) / (bev_configs.lim_y[1] - bev_configs.lim_y[0]) * bev_configs.bev_width
        y = (y - bev_configs.lim_x[0]) / (bev_configs.lim_x[1] - bev_configs.lim_x[0]) * bev_configs.bev_height

        x += bev_configs.bev_width / 2

        z = z - bev_configs.lim_z[0]
        w = w / (bev_configs.lim_y[1] - bev_configs.lim_y[0]) * bev_configs.bev_width
        l = l / (bev_configs.lim_x[1] - bev_configs.lim_x[0]) * bev_configs.bev_height
        yaw = -yaw

        label_box = get_corners(x, y, w , l, yaw)
        if np.all((label_box>= 0) & (label_box <= bev_configs.output_width)):
            #detection_type = detection_mapping[detection.type]
            labels_detections.append(label_box)

    return labels_detections