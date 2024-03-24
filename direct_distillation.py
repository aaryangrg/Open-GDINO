# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
from models.GroundingDINO.groundingdino import GroundingDINOwithEfficientViTBB, build_groundingdino_with_efficientvit_bb
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import importlib

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import  BestMetricHolder
from util.misc import MetricLogger
import util.misc as utils
import datasets
from datasets import bbuild_dataset, bbuild_dataset_custom, get_coco_api_from_dataset
from engine import evaluate, evaluate_custom, train_one_epoch
import torch.distributed as dist

from groundingdino.util.utils import clean_state_dict

#EfficientViT Imports
sys.path.append('/home/aaryang/experiments/Open-GDINO/effvit')
from effvit.efficientvit.clscore.trainer import ClsRunConfig
from effvit.efficientvit.apps import setup
from effvit.efficientvit.clscore.trainer.dino_flexless import GdinoBackboneTrainerNoFlex



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument("--datasets", type=str, required=True, help='path to datasets json')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',help="Train with mixed precision")
    
    parser.add_argument('--distributed', type = bool, default = False)

    parser.add_argument("--config", metavar="FILE", help="config file") # Student Model YAML
    parser.add_argument("--path", type=str, metavar="DIR", help="run directory") # Path for training outs --> checkpoints + logs
    parser.add_argument("--manual_seed", type=int, default=0)
    # parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    # initialization
    parser.add_argument("--rand_init", type=str, default="trunc_normal@0.02")
    parser.add_argument("--last_gamma", type=float, default=0)
    parser.add_argument("--auto_restart_thresh", type=float, default=1.0)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--full_flex_train", type = bool , default = False)
    parser.add_argument("--eval_batch_size", type = int, default = 8)
    parser.add_argument("--effvit_model", type = str, default = None)
    parser.add_argument("--effvit_model_weights_path", type = str, default = None)
    parser.add_argument("--custom_transforms", type = str, default = None)
    parser.add_argument("--custom_res", type = int, default = None)
    parser.add_argument("--kd_loss", type = str, default = "ce")
    parser.add_argument("--pretrained_patch_embed", type = bool, default = False)
    return parser

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args):

    utils.setup_distributed(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))

    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]

    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    # EfficientViT Args parse + setup
    #setup unknown args --> effvit
    config = setup.setup_exp_config(args.config, recursive=True)
    run_config = setup.setup_run_config(config, ClsRunConfig)

    gdino_backbone, _ , _ = build_model_main(args)
    wo_class_error = False
    gdino_backbone.to(device)

    effvit_backbone, criterion, postprocessors = build_groundingdino_with_efficientvit_bb(args, args.effvit_model, args.effvit_model_weights_path)
    effvit_backbone.to("cuda")

    if args.distributed :
        # Make both distributed data parallel for efficient training
        effvit_backbone = torch.nn.parallel.DistributedDataParallel(effvit_backbone, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        effvit_backbone = effvit_backbone.module
        gdino_backbone = torch.nn.parallel.DistributedDataParallel(gdino_backbone, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        gdino_backbone = gdino_backbone.module

    if args.pretrain_model_path: # Load for both models (backbone + efficientViT for validation purposes)
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _ = gdino_backbone.load_state_dict(_tmp_st, strict = False)
        _ = effvit_backbone.load_state_dict(_tmp_st, strict = False)

        # For effivit set backbone[1] to effvit.position_embedding (to use the pretrained weights if trained embeddings)
        effvit_backbone.position_embedding = effvit_backbone.backbone[1]
        
        for param in effvit_backbone.position_embedding.parameters():
            param.requires_grad = False
        
        # For effivit set backbone[0] to effvit.patch_embed (to use the pretrained patch_embeddings)
        if args.pretrained_patch_embed :
            effvit_backbone.patch_embed = effvit_backbone.backbone[0].patch_embed
            for param in effvit_backbone.patch_embed.paraneters():
                param.requires_grad = False

    # Swin-Transformer without Joiner wrapper (skips position embeds)
    gdino_backbone = gdino_backbone.backbone.backbone 
    
    logger.debug("build dataset ... ...")
    dataset_train = bbuild_dataset_custom(image_set='train', args=args, datasetinfo=dataset_meta["train"][0], custom_transforms=args.custom_transforms, custom_res = [args.custom_res])
    dataset_val = bbuild_dataset_custom(image_set='val', args=args, datasetinfo=dataset_meta["val"][0], custom_transforms=args.custom_transforms, custom_res = [args.custom_res])
    logger.debug("build dataset, done.")

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,collate_fn=utils.collate_fn, num_workers=args.num_workers) # default = 4
    data_loader_val = DataLoader(dataset_val, args.eval_batch_size, sampler=sampler_val,drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # default = 8

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    trainer = GdinoBackboneTrainerNoFlex(
        path=args.path,
        effvit_dino=effvit_backbone,
        gdino_backbone = gdino_backbone,
        data_provider=data_loader_train,
        auto_restart_thresh=args.auto_restart_thresh,
        metric_logger = metric_logger,
        train_full_flexible_model = args.full_flex_train,
        fp16_training = args.fp16,
        kd_metric = args.kd_loss
    )

    setup.init_model(
        trainer.network,
        rand_init=args.rand_init,
        last_gamma=args.last_gamma,
    )

    trainer.prep_for_training_custom(run_config, config["ema_decay"], args.fp16)

    base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)
    trainer.train(save_freq=args.save_freq, criterion = criterion, postprocessors = postprocessors, data_loader_val = data_loader_val, base_ds = base_ds, args = args, evaluate_custom = evaluate_custom)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
