import argparse
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import amp

from config import get_config
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from utils.logger import create_logger
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.optimizer import make_optimizer
from utils.scheduler_factory import create_scheduler

if 'RANK' not in os.environ.keys():
    os.environ['RANK'] = '0'
if 'WORLD_SIZE' not in os.environ.keys():
    os.environ['WORLD_SIZE'] = '1'
if 'MASTER_ADDR' not in os.environ.keys():
    os.environ['MASTER_ADDR'] = 'localhost'
if 'MASTER_PORT' not in os.environ.keys():
    os.environ['MASTER_PORT'] = '12345'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_parse():
    parser = argparse.ArgumentParser(description="Attention Enhanced Transformer Network script", add_help=False)
    parser.add_argument('--train_batch', '--train_b', type=int, default=8, help="train batch size for single GPU")
    parser.add_argument('--test_batch', '--test_b', type=int, default=256, help="eval batch size for single GPU")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--config_file", default="./configs/OCC_Duke/AET-Net/FCBAM/SC-AGS.yml", help="path to config file", type=str)

    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--tag', type=str, default='default', help='tag of experiment, suggest options: train or test, default=default')
    # Heat Map Parms
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. Options: mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')


    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    config.freeze()

    return config

def main(config, logger):
    # make datasets
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(config)

    # make model
    model = make_model(config, num_class=num_classes, camera_num=camera_num, view_num=view_num, logger=logger)
    # make loss function
    loss_func, center_criterion = make_loss(config, num_classes=num_classes)
    # make optimizer
    optimizer, optimizer_center = make_optimizer(config, model, center_criterion)
    # make scheduler
    scheduler = create_scheduler(config, optimizer)
    # set model to device
    device = "cuda"
    if device:
        model.to(config.LOCAL_RANK)
        if torch.cuda.device_count() > 1 and config.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], find_unused_parameters=True)

    # Enter interface
    if config.TEST.EVAL_MODE:
        logger.info('Enter Inference Mode...')
        model.load_param(config.TEST.WEIGHT)
        do_eval(model, val_loader, device, num_query, config, logger)
        return

    logger.info("Start training...")
    start_train_time = time.perf_counter()
    checkpoint_period = config.SOLVER.CHECKPOINT_PERIOD
    eval_period = config.SOLVER.EVAL_PERIOD
    epochs = config.SOLVER.MAX_EPOCHS
    scaler = amp.GradScaler()
    for epoch in range(1, epochs + 1):
        model.train()
        train_one_epoch(config, model, train_loader, loss_func, center_criterion, optimizer, optimizer_center, scheduler, scaler, epoch, device, logger)
        # Save Checkpoint
        if epoch % checkpoint_period == 0:
            if config.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(config.OUTPUT_DIR, config.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(config.OUTPUT_DIR, config.MODEL.NAME + '_{}.pth'.format(epoch)))
        # Eval
        if epoch % eval_period == 0:
            do_eval(model, val_loader, device, num_query, config, logger)

def train_one_epoch(config, model, train_loader, loss_func, center_criterion, optimizer, optimizer_center, scheduler, scaler, epoch, device, logger):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    scheduler.step(epoch)
    log_period = config.SOLVER.LOG_PERIOD
    start_epoch_time = time.perf_counter()
    for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img = img.to(device)
        target = vid.to(device)
        target_cam = target_cam.to(device)
        target_view = target_view.to(device)
        with amp.autocast(enabled=True):
            score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
            loss = loss_func(score, feat, target, target_cam, ags=config.MODEL.AGS)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if 'center' in config.MODEL.METRIC_LOSS_TYPE:
            for param in center_criterion.parameters():
                param.grad.data *= (1. / config.SOLVER.CENTER_LOSS_WEIGHT)
            scaler.step(optimizer_center)
            scaler.update()
        if isinstance(score, list):
            acc = (score[0].max(1)[1] == target).float().mean()
        elif isinstance(score, tuple):  # default
            acc = (score[0].max(1)[1] == target).float().mean()
            att_acc = (score[1].max(1)[1] == target).float().mean()
            if acc.item() < att_acc.item():
                acc = att_acc
        else:
            acc = (score.max(1)[1] == target).float().mean()
        loss_meter.update(loss.item(), img.shape[0])
        acc_meter.update(acc, 1)
        torch.cuda.synchronize()

        if (n_iter + 1) % log_period == 0:
            logger.info(f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] "
                        f"Loss: {loss_meter.avg:.3f}, Acc: {acc_meter.avg:.3f}, Base Lr: {scheduler._get_lr(epoch)[0]:.2e}, ")
    end_epoch_time = time.perf_counter()
    time_per_batch = (end_epoch_time - start_epoch_time) / (n_iter + 1)
    logger.info(f"Epoch {epoch} done. Time per batch: {time_per_batch:.3f}[s]. "
                f"Speed: {train_loader.batch_size / time_per_batch:.1f}[samples/s]."
                f"Epoch Time: [{end_epoch_time - start_epoch_time:.2f}]")

def do_eval(model, val_loader, device, num_query, config, logger):
    evaluator = R1_mAP_eval(num_query, config, max_rank=50, feat_norm=config.TEST.FEAT_NORM)
    evaluator.reset()
    model.eval()
    if config.TEST.MAS:
        img_path_list = []
        for n_iter, (img, id, camids, _, viewids, imgpath, val_mask) in enumerate(val_loader):
            with torch.no_grad():
                print("\rProgress: ", end="")
                process_rate = (n_iter + 1) / len(val_loader)
                block_process = len(val_loader) / 20
                num_flag = (n_iter + 1) // block_process
                for index_pro in range(int(num_flag)):
                    print("#", end='')
                print(f"[{round(process_rate * 100, 2)}%]", end='')
                img = img.to(device)
                target_cam = camids.to(device)
                target_view = viewids.to(device)
                feat = model(img, cam_label=target_cam, view_label=target_view)
                evaluator.update((feat, id, camids, val_mask))
                img_path_list.extend(imgpath)
    else:
        for n_iter, (img, id, camids, _, viewids, _, _) in enumerate(val_loader):
            with torch.no_grad():
                print("\rProgress: ", end="")
                process_rate = (n_iter + 1) / len(val_loader)
                block_process = len(val_loader) / 20
                num_flag = (n_iter + 1) // block_process
                for index_pro in range(int(num_flag)):
                    print("#", end='')
                print(f"[{round(process_rate * 100, 2)}%]", end='')
                img = img.to(device)
                target_cam = camids.to(device)
                target_view = viewids.to(device)
                feat = model(img, cam_label=target_cam, view_label=target_view)
                evaluator.update((feat, id, camids))

    if config.TEST.MAS:
        cmc, mAP, MAS = evaluator.compute(model, device, config, img_path_list=img_path_list)
        logger.info("Validation Results:")
        logger.info(f"mAP: {mAP:.1%}")
        logger.info(f"MAS: {MAS*100:.2f}%")
    else:
        cmc, mAP, _ = evaluator.compute(model, device, config)
        logger.info("Validation Results:")
        logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

if __name__ == '__main__':
    # Get Run Config
    config = get_parse()
    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    # else:
    #     rank = -1
    #     world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)

    torch.distributed.init_process_group(backend='gloo', init_method='env://')
    os.environ['CUDA_VISIBLE_DEVICES'] = config.MODEL.DEVICE_ID
    torch.distributed.barrier()
    set_seed(config.SOLVER.SEED)
    # Logger
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT_DIR, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT_DIR, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    logger.info(f"Saving model in the path : {config.OUTPUT_DIR}")
    main(config, logger)
