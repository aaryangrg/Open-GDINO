# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
from .coco import build as build_coco
from .coco import build_custom as build_coco_custom


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def bbuild_dataset(image_set, args, datasetinfo, custom_val_transforms = None, custom_transform_res = None):
    if datasetinfo["dataset_mode"] == 'coco':
        return build_coco(image_set, args, datasetinfo, custom_val_transforms, custom_transform_res)
    if datasetinfo["dataset_mode"] == 'odvg':
        from .odvg import build_odvg
        return build_odvg(image_set, args, datasetinfo)
    raise ValueError(f'dataset {args.dataset_file} not supported')

def bbuild_dataset_custom(image_set, args, datasetinfo, custom_transforms = None, custom_res = None):
    if datasetinfo["dataset_mode"] == 'coco':
        return build_coco_custom(image_set, args, datasetinfo, custom_transforms, custom_res)
    if datasetinfo["dataset_mode"] == 'odvg':
        from .odvg import build_odvg_custom
        return build_odvg_custom(image_set, args, datasetinfo, custom_transforms, custom_res)
    raise ValueError(f'dataset {args.dataset_file} not supported')