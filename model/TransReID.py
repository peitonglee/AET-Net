from IPython import embed
import torch
import torch.nn as nn
import copy

from .backbones.Attention_Module import make_attention_module
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .Init_Model_Parm import weights_init_classifier, weights_init_kaiming

def shuffle_unit(features, shift, group, begin=1):
    # shift: 5
    # group: 2

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

class build_TransReID(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_TransReID, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        local_feature=True, camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        attention_type=cfg.MODEL.Attention_type,
                                                        sem=cfg.MODEL.SEM,
                                                        cem=cfg.MODEL.CEM,
                                                        sem_w=cfg.MODEL.SEM_W,
                                                        cem_w=cfg.MODEL.CEM_W,
                                                        sem_p=cfg.MODEL.SEM_P,
                                                        cem_p=cfg.MODEL.CEM_P,
                                                        ags=cfg.MODEL.AGS)
        self.stride_size = cfg.MODEL.STRIDE_SIZE
        self.img_size = cfg.INPUT.SIZE_TRAIN
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b1_block = copy.deepcopy(block)
        self.b1_norm = copy.deepcopy(layer_norm)

        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.h, self.w = self.img_size[0] // self.stride_size[0], self.img_size[1] // self.stride_size[1]
        embed_dim = 768
        if cfg.MODEL.Attention_type != None:
            attention_model = make_attention_module(name=cfg.MODEL.Attention_type,
                                             img_size=self.img_size,
                                             stride_size=self.stride_size,
                                             embed_dim=embed_dim,
                                             patch_size=self.base.patch_embed.patch_size)

        self.cem = cfg.MODEL.CEM
        self.sem = cfg.MODEL.SEM
        self.sem_p = cfg.MODEL.SEM_P
        self.cem_p = cfg.MODEL.CEM_P
        if self.cem:
            self.cem_w = cfg.MODEL.CEM_W
            self.cem_p = cfg.MODEL.CEM_P
            self.cem_block = attention_model.Channel
            print(f"Channel Weight: {self.cem_w}")
        if self.sem and (self.sem_p == 'after' or self.sem_p == 'all'):
            self.sem_w = cfg.MODEL.SEM_Weight
            self.sem_p = cfg.MODEL.SEM_P
            self.sem_block = attention_model.Spatial
            print(f"Spatial Weight: {self.sem_w}")
            print(f"Spatial Attention Position:{self.sem_p}(Patch_Embed)")
        self.ags = cfg.MODEL.AGS
        self.att_feat_weight = cfg.MODEL.ATT_TRI_WEIGHT
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            if self.ags:
                print(f"Use Attention Loss! Attention Loss Weight: {cfg.MODEL.ATT_TRI_WEIGHT}")
                self.att_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.att_classifier.apply(weights_init_classifier)
                self.att_bottleneck = nn.BatchNorm1d(self.in_planes)
                self.att_bottleneck.bias.requires_grad_(False)
                self.att_bottleneck.apply(weights_init_kaiming)

            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)


        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))

        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        features = self.base(x, cam_label=cam_label, view_label=view_label)
        # global branch
        b1_feat = self.b1_block(features) # [B, 129, 768]
        if self.cem:
            if self.cem_p == 'brfore':
                b1_feat = self.b1_norm(b1_feat)
                c_w = self.cem_block(b1_feat)
                b1_feat = b1_feat * c_w.unsqueeze(dim=1).expand_as(b1_feat) * self.cem_w + b1_feat * (1 - self.cem_w)
            elif self.cem_p == 'after':
                c_w = self.cem_block(b1_feat)
                att_global_feature = b1_feat[:, 0] * c_w * self.cem_w + b1_feat[:, 0] * (1 - self.cem_w)
                att_global_feature = self.b1_norm(att_global_feature)
        b1_feat = self.b1_norm(b1_feat)
        global_feat = b1_feat[:, 0]

        # JPM branch
        # feature_length = features.size(1) - 1
        # patch_length = feature_length // self.divide_length
        patch_length = (self.h // self.divide_length) * self.w
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]


        b1_local_feat = x[:, :patch_length]
        b2_local_feat = x[:, patch_length:patch_length * 2]
        b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        b4_local_feat = x[:, patch_length * 3:]
        if self.sem and (self.sem_p == 'after' or self.sem_p == 'all'):
            local1_s_w = self.sem_block[0](b1_local_feat)
            b1_local_feat = local1_s_w * b1_local_feat * self.sem_w + b1_local_feat * (1 - self.sem_w)

            local2_s_w = self.sem_block[1](b2_local_feat)
            b2_local_feat = local2_s_w * b2_local_feat * self.sem_w + b2_local_feat * (1 - self.sem_w)

            local3_s_w = self.sem_block[2](b3_local_feat)
            b3_local_feat = local3_s_w * b3_local_feat * self.sem_w + b3_local_feat * (1 - self.sem_w)

            local4_s_w = self.sem_block[3](b4_local_feat)
            b4_local_feat = local4_s_w * b4_local_feat * self.sem_w + b4_local_feat * (1 - self.sem_w)

        # lf_1
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)
        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                if self.ags and self.cem:
                    att_feat = self.att_bottleneck(att_global_feature)
                    att_cls_score = self.att_classifier(att_feat)
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            if self.ags and self.cem:
                return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4, att_cls_score], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4, att_global_feature]
            else:
                return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                            cls_score_4
                            ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                                local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                origin_feat = torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)
                if self.cem:
                    att_global_feature = torch.cat((att_global_feature, att_global_feature, att_global_feature, att_global_feature, att_global_feature), dim=1)
                    return att_global_feature * self.att_feat_weight + origin_feat * (1 - self.att_feat_weight)
                else:
                    return origin_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        # param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
