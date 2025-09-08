import argparse
import numpy as np
from thop import profile
import torch
import time
from config import get_config
from model import make_model
from utils.logger import create_logger
from datasets.make_dataloader import make_dataloader


def get_parse():
    parser = argparse.ArgumentParser(description="Attention Enhanced Transformer Network script", add_help=False)
    parser.add_argument('--train_batch', '--train_b', type=int, default=64, help="train batch size for single GPU")
    parser.add_argument('--test_batch', '--test_b', type=int, default=128, help="eval batch size for single GPU")
    # parser.add_argument("--config_file", default="./configs/OCC_Duke/baseline.yml", help="path to config file", type=str)
    parser.add_argument("--config_file", default="./configs/OCC_Duke/TransReID/vit_jpm.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--tag', type=str, help='tag of experiment, suggest options: train or test, default=default')
    # Heat Map Parms
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. Options: mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    config.freeze()
    return config

if __name__ == '__main__':
    cfg = get_parse()
    logger = create_logger(output_dir=cfg.OUTPUT_DIR, name=f"{cfg.MODEL.NAME}")
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(
        cfg, logger=logger)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num, logger=logger)
    model.load_param(cfg.TEST.WEIGHT)
    input_tensor = torch.rand([1, 64, 3, 256, 128])
    start_time = time.perf_counter()
    flops, params = profile(model, input_tensor)
    end_time = time.perf_counter()
    # The denominator is the baseline value
    print(f"time: {(end_time - start_time) / 4.553415299999999}")
    print(f"flops: {flops / 706058747904.0}")
    print(f"params: {params / 85609728.0} ")
    # print(f"time: {end_time - start_time}")
    # print(f"flops: {flops}")
    # print(f"params: {params} ")
