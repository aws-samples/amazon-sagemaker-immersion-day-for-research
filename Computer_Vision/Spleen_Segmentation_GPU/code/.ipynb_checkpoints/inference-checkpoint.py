# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# from monai.utils import first, set_determinism
import numpy
import time
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os, sys, glob, argparse, json, subprocess
import logging
from pathlib import Path
import boto3
# from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

s3_client = boto3.client('s3')
s3 = boto3.resource('s3') # assumes credentials & configuration are handled outside python in .aws directory or environment variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## load model artifact here
def model_fn(model_dir):
#     model = UNet(
#         spatial_dims=3,
#         in_channels=1,
#         out_channels=2,
#         channels=(16, 32, 64, 128, 256),
#         strides=(2, 2, 2, 2),
#         num_res_units=2,
#         norm=Norm.BATCH
#     ).to(device) 

    print("model_dir is", model_dir)
    print("inside model_dir is", os.listdir(model_dir))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
#         model.load_state_dict(torch.load(f,map_location=torch.device('cpu') ))
        model = torch.load(f,map_location=torch.device('cpu') )
        print("model load with cpu!")
    return model.to(device)   


## define data loader for validation dataset
## Notice: val_files including both original image as well as label
## further work should be done in the situation without labels

def get_data_loader(data_files):

    data_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Spacingd(keys=["image"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear")),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            EnsureTyped(keys=["image"]),
        ]
    )

    data_ds = CacheDataset(data=data_files, transform=data_transforms, cache_rate=1.0)
#     data_ds = Dataset(data=data_files, transform=data_transforms)
    data_loader = DataLoader(data_ds, batch_size=1)
    
    return data_loader

## function to download the whole folder
# def download_s3_folder(bucket_name, s3_folder, local_dir=None):
#     """
#     Download the contents of a folder directory
#     Args:
#         bucket_name: the name of the s3 bucket
#         s3_folder: the folder path in the s3 bucket
#         local_dir: a relative or absolute directory path in the local file system
#     """
#     bucket = s3.Bucket(bucket_name)
#     for obj in bucket.objects.filter(Prefix=s3_folder):
#         target = obj.key if local_dir is None \
#             else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
#         if not os.path.exists(os.path.dirname(target)):
#             os.makedirs(os.path.dirname(target))
#         if obj.key[-1] == '/':
#             continue
#         bucket.download_file(obj.key, target)
        
#     return


def input_fn(serialized_input_data, content_type):
    s3_client = boto3.client('s3')
    s3 = boto3.resource('s3') # assumes credentials & configuration are handled outside python in .aws directory or environment variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Received request of type:{content_type}")
    
    print("serialized_input_data is---", serialized_input_data)
    if content_type == 'application/json':
        
        data = json.loads(serialized_input_data)
        
        bucketname=data['bucket']
        s3_folder=data['key']## prefix with all the image files as well as labelings
        filestring=data["file"]
        
        if filestring[-2:]=="gz":
            file=filestring
        else:
            file=".".join(filestring.split(".")[:-1])
        
        nslice=int(data["nslice"])
        
        local_dir = "tmp"
        
        try:
            os.makedirs(local_dir)
            print("Directory '%s' created" %local_dir)
        except:
            print("Directory '%s' was not created" %local_dir)
            
        source = os.path.join(s3_folder, file)
        target = os.path.join(local_dir, file)
        bucket = s3.Bucket(bucketname)
        bucket.download_file(source, target)
        print(f'Download {source} to {target} finished!')

        ## Download the folder from s3         
        # download into local folder
#         download_s3_folder(bucket, s3_folder, local_dir=local_dir)
#         print("Downloaded files from S3. Bucket:" , bucket, " Key: ",s3_folder)

        print('Start to inference for the file >>> ', file)
        
        images = [0]
        images[0] = target
        data_dicts = [{"image": image_name} for image_name in images]
        data_loader = get_data_loader(data_dicts)
        print('get_data_loader finished!')
        
        for i, data_l in enumerate(data_loader):
                pred_input = data_l["image"].to(device)
        shutil. rmtree(local_dir) ## delete all the files after loading

        
        return pred_input, nslice, bucketname

    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


def predict_fn(input_data, model):
    print('Got input Data: {}'.format(input_data))
    print("input_fn in predict:",input_data)
    infer_loader, nslice, bucketname = input_data
    
    print("bucket name is :,", bucketname)
    s3_client = boto3.client('s3')
    s3 = boto3.resource('s3')
    bucket_local = s3.Bucket(bucketname)
    print("bucket_local name is :,", bucket_local)
    roi_size = (160, 160, 160)
    sw_batch_size = 4
    
    test_output = sliding_window_inference(infer_loader, roi_size, sw_batch_size, model)
    
    ## test for output size start here
    if(nslice<100):
        print("output a single slice :", nslice)
        infer_output = torch.argmax(test_output, dim=1).detach().cpu()[0, :, :, nslice].tolist()
    elif((nslice>=100)&(nslice<200)):
        print("output 10 slices :", nslice)
        infer_output = torch.argmax(test_output, dim=1).detach().cpu()[0, :, :, 70:81].tolist()
    else:
        print("output all slides")
        infer_output = torch.argmax(test_output, dim=1).detach().cpu()[0, :, :, :].tolist()
    
#     for i, val_data in enumerate(input_data):
#     val_outputs = sliding_window_inference(
#         val_data, roi_size, sw_batch_size, model
#     )
#     val_list.append(torch.argmax(
#         val_outputs, dim=1).detach().cpu()[0, :, :, 80].tolist())
    print("finished pred_fn!")
    
    ## save the results in S3
    pred_json = {"pred:": infer_output}
    json_string = json.dumps(pred_json)

    ext=str(time.time()) ## current timestamp
    name="results_slides"+ext+".json"
    print("file name after prediction is:", name)
    ## save as json file in the container
    with open(name, 'w') as outfile:
        outfile.write(json_string)
    
    ##upload the results to S3
    prefix='inference_output'
    s3_path=bucket_local.upload_file(name, prefix+'/'+name)
    s3_path=os.path.join(f"S3://{bucketname}",prefix,name)
    os.remove(name) ## delete the file after uploading
    print("s3_path after uploading is", s3_path)
    return s3_path,infer_output,nslice


def output_fn(prediction_output, content_type):
    
    #print("inside output_fn with length ==", len(prediction_output))


    s3_path, tensor_output,nslice=prediction_output
    
    print("s3_path after uploading is", s3_path)


    try:
        pred_json = {"s3": s3_path}
        if(nslice<100):
            pred_json = {"s3_path": s3_path,"pred": tensor_output }
        return pred_json
    except:
        raise Exception('Requested unsupported ContentType: ' + content_type)

