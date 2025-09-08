import argparse
import os.path
import sys
from config import get_config
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
from time import sleep
from IPython import embed

from model import make_model
from datasets import make_dataloader
from utils.vit_rollout import VITAttentionRollout
starytegy_factory = {
    'configs': [],
    'model_type': []
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument("--config_file", default="./configs/OCC_D/baseline.yml", help="path to config file", type=str)
    parser.add_argument('--image_path', type=str, default='./imgs', help='Input image path or floder path')
    parser.add_argument('--OUTPUT_DIR', type=str, default='./imgs/result', help='Output heat map path')
    parser.add_argument('--Model_Type', type=str, default='stride11_Relation_c', help="The model's tag label")
    parser.add_argument('--head_fusion', type=str, default='max', help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9, help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--show', type=bool, default=False, help='show the result of heat map')
    parser.add_argument('--category_index', type=int, default=None, help='The category index for gradient rollout')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = get_config(args)
    config.freeze()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")
    return args, config

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def visual(args, img_name, transform, model, local=False):
    img = Image.open(img_name)
    img = img.resize((128, 256))
    input_tensor = transform(img).unsqueeze(0)
    if args.use_cuda:
        input_tensor = input_tensor.cuda()
    attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
        discard_ratio=args.discard_ratio, local=local)
    mask = attention_rollout(input_tensor)
    name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
    save_path = os.path.join(args.OUTPUT_DIR, args.Model_Type)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_name = os.path.join(save_path, img_name.split('\\')[-1])
    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)

    # cv2.imshow("Input Image", np_img)
    if args.show:
        cv2.imshow(name, mask)
    cv2.imwrite(save_name, mask)
    print(f'\tSave heat map image to {save_name}', end='')


if __name__ == '__main__':
    args, cfg = get_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    if cfg.MODEL.NAME and cfg.MODEL.JPM:
        local = True
    if args.use_cuda:
        model = model.cuda()
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    if not os.path.exists(args.image_path):
        raise ValueError(f'{args.image_path} is not exist!')
    if os.path.isdir(args.image_path):
        for idx, img_name in enumerate(os.listdir(args.image_path)):
            sleep(0.5)
            print("\rProgress: ", end="")
            for i in range(idx):
                print("##", end='')
            print(f"[{(idx+1)/len(os.listdir(args.image_path)) * 100}%]", end='')
            img_path = os.path.join(args.image_path, img_name)
            visual(args, img_path, transform, model, local=local)
    else:
        visual(args, args.image_path, transform, model, local=local)
