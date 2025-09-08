import torch
import numpy as np
import os
import cv2
from utils.reranking import re_ranking
from IPython import embed
from PIL import Image
from torchvision import transforms
from utils.vit_rollout import VITAttentionRollout


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # remove = (g_pids[order] == q_pid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def calc_MAS(masks, heat_maps, thegema=0.1):
    print('\n=> Computing Model Attention Score(MAS)...')
    mas = 0
    for n_iter in range(masks.shape[0]):
        print("\rMAS Progress: ", end="")
        process_rate = (n_iter + 1) / masks.shape[0]
        block_process = masks.shape[0] // 40
        num_flag = (n_iter + 1) // block_process
        for index_pro in range(int(num_flag)):
            print("=", end='')
        print(f"[{round(process_rate * 100, 2)}%]", end='')
        mask = masks[n_iter]
        mask = (mask == 0) * thegema + mask
        heat_map = heat_maps[n_iter]
        heat_map_sum = heat_map.sum()
        heat_map = heat_map / heat_map_sum
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

def get_heat_map(model, image_list, device, cfg):
    root = os.path.join(cfg.OUTPUT_DIR, "heat_map")
    heat_map_list = []
    local = (cfg.MODEL.NAME == 'transreid') and (cfg.MODEL.JPM)
    attention_rollout = VITAttentionRollout(model, head_fusion=cfg.HEATMAP.HEAD_FUSION,
                                            discard_ratio=cfg.HEATMAP.DISCARD_RATIO, local=local, cfg=cfg)
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    save_heat_map = cfg.HEATMAP.SAVE
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
        if cfg.HEATMAP.SAVE:
            # different OS hanve different split way
            #
            current_img_type = image_path.split("/")[-2]
            current_img_name = image_path.split("/")[-1]
            # current_img_type = image_path.split("/")[-2]
            # current_img_name = image_path.split("/")[-1]
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
    return torch.from_numpy(np.array(heat_map_list))

class R1_mAP_eval():
    def __init__(self, num_query, cfg, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.MAS = cfg.TEST.MAS
        self.reranking = reranking
        self.thegema = cfg.TEST.MASK_THEGEMA

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.mask = []

    def update(self, output):  # called once for each batch
        if self.model_name == 'transreid':
            if self.MAS:
                feat, pid, camid, mask = output
                self.mask.append(mask)
            else:
                feat, pid, camid = output

        elif self.model_name == 'AET-Net':
            if self.MAS:
                feat, pid, camid, mask = output
                self.mask.append(mask)  # torch.tensor
            else:
                feat, pid, camid = output
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.feats.append(feat.cpu())   # torch.tensor


    def compute(self, model, device, cfg, img_path_list=None):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.MAS:
            masks = torch.cat(self.mask, dim=0)
            heat_maps = get_heat_map(model, img_path_list, device, cfg)
            MAS = calc_MAS(masks, heat_maps, thegema=self.thegema)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        if self.MAS:
            return cmc, mAP, MAS
        else:
            return cmc, mAP, None



