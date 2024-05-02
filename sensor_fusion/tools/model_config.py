from easydict import EasyDict as edict
import os
import numpy as np
import torch




def create_model_config(model_name='darknet', configs=None):

    if configs==None:
        configs = edict()  

    curr_path = os.path.dirname(os.path.realpath(__file__))   
    
    if model_name == "darknet":
        configs.model_path = os.path.join(curr_path, 'tools', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False

    elif model_name == 'fpn_resnet':
        print("student task ID_S3_EX1-3")
        configs.arch = 'fpn_resnet'
        configs.model_path = os.path.join(curr_path, 'resnet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'fpn_resnet_18_epoch_300.pth')
        configs.K = 50
        configs.batch_size = 1
        configs.peak_thresh = 0.2
        configs.conf_thresh = configs.peak_thresh 

        configs.pin_memory = True
        configs.distributed = False  

        configs.input_size = (608, 608)
        configs.hm_size = (152, 152)
        configs.down_ratio = 4
        configs.max_objects = 50

        configs.imagenet_pretrained = False
        configs.head_conv = 64
        configs.num_classes = 3
        configs.num_center_offset = 2
        configs.num_z = 1
        configs.num_dim = 3
        configs.num_direction = 2  

        configs.heads = {
            'hm_cen': configs.num_classes,
            'cen_offset': configs.num_center_offset,
            'direction': configs.num_direction,
            'z_coor': configs.num_z,
            'dim': configs.num_dim
        }
        configs.num_input_features = 4
  

    else:
        raise ValueError("Error: Invalid model name")

    configs.no_cuda = True 
    configs.gpu_idx = 0 
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    configs.min_iou = 0.5

    return configs