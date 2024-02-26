# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
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
from datasets import bbuild_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
import torch.distributed as dist

from groundingdino.util.utils import clean_state_dict

#EfficientViT Imports
sys.path.append('/home/aaryang/experiments/Open-GDINO/effvit')
from effvit.efficientvit.models.efficientvit.dino_backbone import flexible_efficientvit_backbone_swin_t_224_1k
from effvit.efficientvit.clscore.trainer import ClsRunConfig
from effvit.efficientvit.clscore.trainer.mutual_trainer import ClsMutualTrainer
from effvit.efficientvit.models.nn.drop import apply_drop_func
from effvit.efficientvit.apps.utils import dump_config, parse_unknown_args
from effvit.efficientvit.apps import setup
from effvit.efficientvit.clscore.trainer.gdino_backbone import GdinoBackboneTrainer



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

    # if args.rank == 0:
    #     save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
    #     cfg.dump(save_cfg_path)
    #     save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
    #     with open(save_json_path, 'w') as f:
    #         json.dump(vars(args), f, indent=2)

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

    # if args.rank == 0:
    #     save_json_path = os.path.join(args.output_dir, "config_args_all.json")
    #     with open(save_json_path, 'w') as f:
    #         json.dump(vars(args), f, indent=2)
        # logger.info("Full config saved to {}".format(save_json_path))

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

    # Extract LR scheduler + Optimizer from this

    logger.debug("build model ... ...")
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    logger.debug("build model, done.")

    pytorch_total = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print("Swin backbone params : ", pytorch_total)

    effvit_backbone = flexible_efficientvit_backbone_swin_t_224_1k()
    effvit_backbone.to("cuda")
    pytorch_total_params = sum(p.numel() for p in effvit_backbone.parameters() if p.requires_grad)
    print("Backbone params : ", pytorch_total_params)

    # make effvit_backbone data parallel as well as the main model (set to eval)
    model_without_ddp = model

    if args.distributed :

        effvit_backbone = torch.nn.parallel.DistributedDataParallel(effvit_backbone, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        effvit_backbone = effvit_backbone.module
        # effvit_backbone = torch.nn.DataParallel(effvit_backbone)
        # effvit_backbone = effvit_backbone.module
    
    logger.debug("build dataset ... ...")
    dataset_train = bbuild_dataset(image_set='train', args=args, datasetinfo=dataset_meta["train"][0])
    dataset_val = bbuild_dataset(image_set='val', args=args, datasetinfo=dataset_meta["val"][0])
    logger.debug("build dataset, done.")

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    print("batch_size : ", args.batch_size)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 4, sampler=sampler_val,drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    trainer = GdinoBackboneTrainer(
        path=args.path,
        vit_backbone=effvit_backbone,
        dino_backbone = model_without_ddp,
        data_provider=data_loader_train,
        auto_restart_thresh=args.auto_restart_thresh,
        metric_logger = metric_logger,
        train_full_flexible_model = args.full_flex_train
    )

    setup.init_model(
        trainer.network,
        rand_init=args.rand_init,
        last_gamma=args.last_gamma,
    )

    trainer.prep_for_training_custom(run_config, config["ema_decay"], args.fp16)

    # Extract LR scheduler + Optimizer + scaler??

    base_ds = get_coco_api_from_dataset(dataset_val)

    # Initial pre-trained weight load

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(clean_state_dict(checkpoint['model']),strict=False)

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')

    if  args.pretrain_model_path:
        print("loading original model weights")
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
        print("Cleaning state dict")
        # logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

    trainer.train(save_freq=args.save_freq)
 
    
    # if args.eval:
    #     os.environ['EVAL_FLAG'] = 'TRUE'
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
    #     if args.output_dir:
    #         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

    #     log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
    #     if args.output_dir and utils.is_main_process():
    #         with (output_dir / "log.txt").open("a") as f:
    #             f.write(json.dumps(log_stats) + "\n")

    #     return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
