### manually - For now copy the conf file from the spleen segmentation ### 
### manually - copy the pretrained model to where you want the bundle to be created ### 


import tempfile
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Any, Mapping, Hashable

import monai
from monai.config import print_config
from monai.utils import first
from monai.config import KeysCollection
from monai.data import Dataset, ArrayDataset, create_test_image_3d, DataLoader
from monai.transforms import (
    Transform,
    MapTransform,
    Randomizable,
    AddChannel,
    AddChanneld,
    Compose,
    LoadImage,
    LoadImaged,
    Lambda,
    Lambdad,
    RandSpatialCrop,
    RandSpatialCropd,
    ToTensor,
    ToTensord,
    Orientation, 
    Rotate
)
print_config()

import monailabel
import os 
import sys 

from monai.bundle import ConfigParser
import json 
import gdown 
import wget 

## Create local directories to hold the bundle info ### 
main_dir = "D:\deepa\Project_week\PW38\ct_full_seg_bundle"
model_path_github = "https://github.com/deepakri201/monai_app_to_bundle/tree/main/pretrained_models/segmentation_full_ct.pt"

config_dir = os.path.join(main_dir, 'configs')
model_dir = os.path.join(main_dir, 'models')
data_dir = os.path.join(main_dir,"data")
data_dir_Ts = os.path.join(data_dir, "imagesTs")

if not os.path.isdir(main_dir):
  os.mkdir(main_dir)
if not os.path.isdir(main_dir):
  os.mkdir(main_dir)
if not os.path.isdir(config_dir):
  os.mkdir(config_dir)
if not os.path.isdir(model_dir):
  os.mkdir(model_dir)
if not os.path.isdir(data_dir):
  os.mkdir(data_dir)
if not os.path.isdir(data_dir_Ts):
  os.mkdir(data_dir_Ts)


###### For later, copy the file from github ###### 
### Copy the pretrained model from the google drive link to the directory models, rename as model.pt ### 
# https://drive.google.com/file/d/1_ZvyzpO8MMpFTph9XCcnQmhErSyA5QR7/view?usp=share_link

# url = 'https://drive.google.com/file/d/1_ZvyzpO8MMpFTph9XCcnQmhErSyA5QR7/view?usp=share_link'
# output = 'pretrained_models.zip'
# gdown.download(url, output, quiet=False)

# Copy the pretrained model from github to the directory we are creating the bundle in 

# filename = wget.download(model_path_github, out=model_dir)

# model_path = os.path.join(model_dir, "model.pt")
#
# print('filename: ' + str(filename))
# print('model_path: ' + str(model_path))
#
# os.rename(os.path.join(model_dir,filename), model_path)

# !wget $model_path_github 
# !mv "/content/radiology_segmentation_segresnet_localization_spine.pt" $model_path
###########################################################

############################################################

#### Create the metajson file ### 


metajson_dict = {
    "schema": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/meta_schema_20220324.json",
    "version": "0.1.1",
    "changelog": {
        "0.1.1": "disable image saving during evaluation",
        "0.1.0": "complete the model package",
        "0.0.1": "initialize the model package structure"
    },
    "monai_version": "0.9.0",
    "pytorch_version": "1.10.0",
    "numpy_version": "1.21.2",
    "optional_packages_version": {
        "nibabel": "3.2.1",
        "pytorch-ignite": "0.4.8"
    },
    "task": "Decathlon spleen segmentation",
    "description": "A pre-trained model for volumetric (3D) segmentation of the spleen from CT image",
    "authors": "MONAI team",
    "copyright": "Copyright (c) MONAI Consortium",
    "data_source": "Task09_Spleen.tar from http://medicaldecathlon.com/",
    "data_type": "nibabel",
    "image_classes": "single channel data, intensity scaled to [0, 1]",
    "label_classes": "single channel data, 1 is spleen, 0 is everything else",
    "pred_classes": "2 channels OneHot data, channel 1 is spleen, channel 0 is background",
    "eval_metrics": {
        "mean_dice": 0.96
    },
    "intended_use": "This is an example, not to be used for diagnostic purposes",
    "references": [
        "Xia, Yingda, et al. '3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training. arXiv preprint arXiv:1811.12506 (2018). https://arxiv.org/abs/1811.12506.",
        "Kerfoot E., Clough J., Oksuz I., Lee J., King A.P., Schnabel J.A. (2019) Left-Ventricle Quantification Using Residual U-Net. In: Pop M. et al. (eds) Statistical Atlases and Computational Models of the Heart. Atrial Segmentation and LV Quantification Challenges. STACOM 2018. Lecture Notes in Computer Science, vol 11395. Springer, Cham. https://doi.org/10.1007/978-3-030-12029-0_40"
    ],
    "network_data_format": {
        "inputs": {
            "image": {
                "type": "image",
                "format": "hounsfield",
                "modality": "CT",
                "num_channels": 1, # check this
                "spatial_shape": [
                    96,
                    96,
                    96
                ],
                "dtype": "float32",
                "value_range": [
                    0,
                    1
                ],
                "is_patch_data": True,
                "channel_def": {
                    "0": "image"
                }
            }
        },
        "outputs": {
            "pred": {
                "type": "image",
                "format": "segmentation",
                "num_channels": 104, # can get this automatically - check if 104 to 105 including background
                # "num_channels": 2, 
                "spatial_shape": [
                    96,
                    96,
                    96
                ],
                "dtype": "float32",
                "value_range": [
                    0,
                    1
                ],
                "is_patch_data": True,
                # "channel_def": { # WHY 25. should be 1 for the first spine localization step? 
                #     "0": "background",
                #     "1": "spine"
                # }
                "channel_def": {
                "spleen": 1,
                "kidney_right": 2,
                "kidney_left": 3,
                "gallbladder": 4,
                "liver": 5,
                "stomach": 6,
                "aorta": 7,
                "inferior_vena_cava": 8,
                "portal_vein_and_splenic_vein": 9,
                "pancreas": 10,
                "adrenal_gland_right": 11,
                "adrenal_gland_left": 12,
                "lung_upper_lobe_left": 13,
                "lung_lower_lobe_left": 14,
                "lung_upper_lobe_right": 15,
                "lung_middle_lobe_right": 16,
                "lung_lower_lobe_right": 17,
                "vertebrae_L5": 18,
                "vertebrae_L4": 19,
                "vertebrae_L3": 20,
                "vertebrae_L2": 21,
                "vertebrae_L1": 22,
                "vertebrae_T12": 23,
                "vertebrae_T11": 24,
                "vertebrae_T10": 25,
                "vertebrae_T9": 26,
                "vertebrae_T8": 27,
                "vertebrae_T7": 28,
                "vertebrae_T6": 29,
                "vertebrae_T5": 30,
                "vertebrae_T4": 31,
                "vertebrae_T3": 32,
                "vertebrae_T2": 33,
                "vertebrae_T1": 34,
                "vertebrae_C7": 35,
                "vertebrae_C6": 36,
                "vertebrae_C5": 37,
                "vertebrae_C4": 38,
                "vertebrae_C3": 39,
                "vertebrae_C2": 40,
                "vertebrae_C1": 41,
                "esophagus": 42,
                "trachea": 43,
                "heart_myocardium": 44,
                "heart_atrium_left": 45,
                "heart_ventricle_left": 46,
                "heart_atrium_right": 47,
                "heart_ventricle_right": 48,
                "pulmonary_artery": 49,
                "brain": 50,
                "iliac_artery_left": 51,
                "iliac_artery_right": 52,
                "iliac_vena_left": 53,
                "iliac_vena_right": 54,
                "small_bowel": 55,
                "duodenum": 56,
                "colon": 57,
                "rib_left_1": 58,
                "rib_left_2": 59,
                "rib_left_3": 60,
                "rib_left_4": 61,
                "rib_left_5": 62,
                "rib_left_6": 63,
                "rib_left_7": 64,
                "rib_left_8": 65,
                "rib_left_9": 66,
                "rib_left_10": 67,
                "rib_left_11": 68,
                "rib_left_12": 69,
                "rib_right_1": 70,
                "rib_right_2": 71,
                "rib_right_3": 72,
                "rib_right_4": 73,
                "rib_right_5": 74,
                "rib_right_6": 75,
                "rib_right_7": 76,
                "rib_right_8": 77,
                "rib_right_9": 78,
                "rib_right_10": 79,
                "rib_right_11": 80,
                "rib_right_12": 81,
                "humerus_left": 82,
                "humerus_right": 83,
                "scapula_left": 84,
                "scapula_right": 85,
                "clavicula_left": 86,
                "clavicula_right": 87,
                "femur_left": 88,
                "femur_right": 89,
                "hip_left": 90,
                "hip_right": 91,
                "sacrum": 92,
                "face": 93,
                "gluteus_maximus_left": 94,
                "gluteus_maximus_right": 95,
                "gluteus_medius_left": 96,
                "gluteus_medius_right": 97,
                "gluteus_minimus_left": 98,
                "gluteus_minimus_right": 99,
                "autochthon_left": 100,
                "autochthon_right": 101,
                "iliopsoas_left": 102,
                "iliopsoas_right": 103,
                "urinary_bladder": 104,
        }
            }
        }
    }
}

config = ConfigParser()
metajson_filename = os.path.join(config_dir, 'metadata.json')
config.export_config_file(metajson_dict, metajson_filename, fmt="json")

### Create inference.json file ### 


inference_dict = {
    "imports": [
        "$import glob",
        "$import os",
        "$import sys",
    ],
    "bundle_root": "", 
    "output_dir": "$@bundle_root + '/eval'",
    "dataset_dir": "",
    "datalist": "$list(sorted(glob.glob(@dataset_dir + '/imagesTs/*.nii.gz')))",
    "device": "$torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')",
    ### I modified the network_def below ### 
    "network_def": { # https://github.com/Project-MONAI/MONAILabel/blob/main/sample-apps/radiology/lib/configs/localization_spine.py
        "_target_": "SegResNet",
        "spatial_dims": 3,
        "init_filters": 32,
        "in_channels": 1,
        "out_channels": 105,  # can get this automatically later 
        # "out_channels": 2,
        "dropout_prob": 0.2,
        "blocks_down": (1, 2, 2, 4),
        "blocks_up": (1, 1, 1)
    },


    "network": "$@network_def.to(@device)",
    ### I modified the preprocessing below ### # pre_transforms from https://github.com/Project-MONAI/MONAILabel/blob/main/sample-apps/radiology/lib/infers/localization_spine.py 
    "preprocessing": {
        "_target_": "Compose",
        "transforms": [
            {
                "_target_": "LoadImaged", # LoadImaged(keys="image"),
                "keys": "image"
            },
            {
                "_target_": "EnsureTyped", # EnsureTyped(keys="image", device=data.get("device") if data else None),
                "keys": "image"
            },
            {
                "_target_": "EnsureChannelFirstd", #  EnsureChannelFirstd(keys="image"),
                "keys": "image"
            },
            {
                "_target_": "Orientationd", # Orientationd(keys="image", axcodes="RAS"),
                "keys": "image",
                "axcodes": "RAS"
            },
            {
                "_target_": "Spacingd", # Spacingd(keys="image", pixdim=self.target_spacing, allow_missing_keys=True),
                "keys": "image",
                "pixdim": [
                    1.5,
                    1.5,
                    1.5
                ],
                "mode": "bilinear"
            },
            {
                "_target_": "Spacingd", # NormalizeIntensityd(keys="image", nonzero=True),
                "keys": "image",
                "pixdim": [
                    1.5,
                    1.5,
                    1.5
                ],
                "mode": "bilinear"
            },
            # ScaleIntensityRanged(keys="image", a_min=-1000, a_max=1900, b_min=0.0, b_max=1.0, clip=True),
            {
                "_target_": "GaussianSmoothd", # GaussianSmoothd(keys="image", sigma=0.4),
                "keys": "image",
                "sigma": 0.4 
            },
            {
                "_target_": "ScaleIntensityd", # ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
                "keys": "image",
                "minv": -1.0, 
                "maxv": 1.0
            }, 
        ]

},
    "dataset": {
        "_target_": "Dataset",
        "data": "$[{'image': i} for i in @datalist]",
        "transform": "@preprocessing"
    },
    ### Will need to find the batch_size and num_workers ### 
    "dataloader": {
        "_target_": "DataLoader",
        "dataset": "@dataset",
        "batch_size": 1,  # CHECK THIS!
        "shuffle": False,
        "num_workers": 4   # CHECK THIS!
    },
    ### I modified the inferer below ### inferer from https://github.com/Project-MONAI/MONAILabel/blob/main/sample-apps/radiology/lib/infers/localization_spine.py 
    "inferer": {
        "_target_": "SlidingWindowInferer", # SlidingWindowInferer(roi_size=self.roi_size, sw_batch_size=2, overlap=0.4, padding_mode="replicate", mode="gaussian")
        "roi_size": [ # roi_size from here # https://github.com/Project-MONAI/MONAILabel/blob/main/sample-apps/radiology/lib/configs/localization_spine.py
            96,
            96,
            96
        ],
        "sw_batch_size": 2,
        "overlap": 0.4,
        "padding_mode": "replicate", 
        "mode": "gaussian"
    },
    ### Need to modify the below ### post_transforms from https://github.com/Project-MONAI/MONAILabel/blob/main/sample-apps/radiology/lib/infers/localization_spine.py 
    "postprocessing": {
        "_target_": "Compose",
        "transforms": [
           {
                "_target_": "EnsureTyped", # EnsureTyped(keys="pred", device=torch.device("cpu")), # Use this in case of using GPU smaller than 24GB
                "keys": "image",
                # "device": "$torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"
            },
           {
                "_target_": "Activationsd", # Activationsd(keys="pred", softmax=True), # https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/post/dictionary.py - do I need to specify the other keys? keep defaults.
                "keys": "pred",
                "softmax": True
            },
           {
                "_target_": "AsDiscreted", # AsDiscreted(keys="pred", argmax=True), # https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/post/dictionary.py - do I need to specify the other keys? keep defaults.
                "keys": "pred",
                "argmax": True
            },
            {
                "_target_": "KeepLargestConnectedComponentd", # KeepLargestConnectedComponentd(keys="pred"), # https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/post/dictionary.py  - do I need to specify the other keys? keep defaults.
                "keys": "pred"
            },
            ### Restored is from monailabel - from monailabel.transform.post import Restored ### 
            # {
            #     "_target_": "Restored", # Restored(keys="pred", ref_image="image"), # https://github.com/Project-MONAI/MONAILabel/blob/main/monailabel/transform/post.py # keep defaults 
            #     "keys": "pred",
            #     "ref_image": "image"
            # },
            {
                "_target_": "EnsureChannelFirstd", # EnsureChannelFirstd(keys="pred"), # https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/post/dictionary.py  - do I need to specify the other keys? keep defaults.
                "keys": "pred"
            },
            ### RestoreOrientationd is from from lib.transforms.transforms import RestoreOrientationd  ### 
            # RestoreOrientationd(keys="pred", ref_image="image"),
            # Do we need the above?? 
            ### I added SaveImaged below ### 
            {
                "_target_": "SaveImaged",
                "keys": "pred",
                # "meta_keys": "pred_meta_dict",
                "output_dir": "@output_dir"
            }
        ]
    },
    "handlers": [
        {
            "_target_": "CheckpointLoader",
            "load_path": "$@bundle_root + '/models/model.pt'",
            "load_dict": {
                "model": "@network"
            }
        },
        {
            "_target_": "StatsHandler",
            "iteration_log": False
        }
    ],
    "evaluator": {
        "_target_": "SupervisedEvaluator",
        "device": "@device",
        "val_data_loader": "@dataloader",
        "network": "@network",
        "inferer": "@inferer",
        "postprocessing": "@postprocessing",
        "val_handlers": "@handlers",
        "amp": True
    },
    "evaluating": [
        "$setattr(torch.backends.cudnn, 'benchmark', True)",
        "$@evaluator.run()"
    ]
}

config = ConfigParser()
inference_filename = os.path.join(config_dir, 'inference.json')
# Set the bundle_root
config["bundle_root"] = main_dir
# Set the dataset_dir
config["dataset_dir"] = data_dir

config_dict = config.get() 
inference_json = os.path.join(config_dir, "inference.json")
config.export_config_file(config_dict, inference_json, fmt="json")

# config.export_config_file(inference_dict, inference_filename, fmt="json")

