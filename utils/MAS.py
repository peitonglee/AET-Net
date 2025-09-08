import torch
import numpy as np
import os
import cv2
from utils.reranking import re_ranking
from IPython import embed
from PIL import Image
from torchvision import transforms
from utils.vit_rollout import VITAttentionRollout


def calc_MAS(masks, heat_maps, thegema=0.1):
    print('\n=> Computing Model Attention Score(MAS)...')
    mas = 0
    num_imgs = masks.shape[0]
    for n_iter in range(masks.shape[0]):
        print("\rMAS Progress: ", end="")
        process_rate = (n_iter + 1) / masks.shape[0]
        block_process = masks.shape[0] // 40
        num_flag = (n_iter + 1) // block_process
        for index_pro in range(int(num_flag)):
            print("=", end='')
        print(f"[{round(process_rate * 100, 2)}%]", end='')
        heat_map = heat_maps[n_iter]
        if heat_map.sum() == 0:
            mas += 0.3
            continue
        mask = masks[n_iter]
        mask = (mask == 0) * thegema + mask
        heat_map_sum = heat_map.sum()
        for i in range(heat_map.shape[0]):
            for j in range(heat_map.shape[1]):
                if heat_map[i, j] / heat_map_sum < (1/(heat_map.shape[0] * heat_map.shape[1])) * 1.0:
                    heat_map[i, j] = 0
        if heat_map.sum() == 0:
            mas += 0.3
            continue
        heat_map = heat_map / heat_map.sum()
        mas += (heat_map * mask).sum().item()
    print()
    res_mas = mas / (masks.shape[0])
    return res_mas

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def get_heat_map(model, image_list, device, args, cfg):

    root = os.path.join(cfg.OUTPUT_DIR, "heat_map")
    if not os.path.exists(root):
        os.makedirs(root)
    if os.path.exists(root + '/heat_map.npy'):
        return  torch.from_numpy(np.load(root + '/heat_map.npy'))
    heat_map_list = []
    attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
                                            discard_ratio=args.discard_ratio, local=cfg.MODEL.JPM,
                                            cfg=cfg)
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    save_heat_map = cfg.TEST.SAVE_HEAT_MAP
    print()
    print(f"Save Heat Map: {save_heat_map}")
    if save_heat_map:
        print(f"Save Heat Map Path: '{root}'")
    for index, image_path in enumerate(image_list):
        print("\rComputer Heat Map Progress: ", end="")
        process_rate = (index + 1) /len(image_list)
        block_process = len(image_list) // 20
        num_flag = (index + 1) // block_process
        for index_pro in range(int(num_flag)):
            print("=", end='')
        print(f"[{round(process_rate * 100, 2)}%]", end='')
        img = Image.open(image_path)
        img = img.resize((128, 256))
        input_tensor = transform(img).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        mask = attention_rollout(input_tensor)
        np_img = np.array(img)[:, :, ::-1]
        if cfg.TEST.SAVE_HEAT_MAP:
            current_img_type = image_path.split("/")[-2]
            current_img_name = image_path.split("/")[-1]
            save_path = os.path.join(root, current_img_type)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = os.path.join(save_path, current_img_name)
            heat_map = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
            heat_image = show_mask_on_image(np_img, heat_map)
            cv2.imwrite(save_name, heat_image)
        else:
            heat_map = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        heat_map_list.append(heat_map)
    heat_map = np.array(heat_map_list)
    np.save(root + '/heat_map.npy', heat_map)
    return torch.from_numpy(heat_map)

class MAS():
    def __init__(self, mask_thegema=0.1):
        super(MAS, self).__init__()
        self.thegema = mask_thegema

    def reset(self):
        self.mask = []

    def update(self, mask):  # called once for each batch
        self.mask.append(mask)  # torch.tensor

    def compute(self, model, img_path_list, device, args, cfg):  # called after each epoch
        masks = torch.cat(self.mask, dim=0)
        heat_maps = get_heat_map(model, img_path_list, device, args, cfg)
        MAS = calc_MAS(masks, heat_maps, thegema=self.thegema)
        return MAS



