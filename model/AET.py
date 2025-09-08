import torch
import torch.nn as nn
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .Init_Model_Parm import weights_init_classifier, weights_init_kaiming

class build_AET(nn.Module):
    def __init__(self, num_classes, cfg, factory):
        super(build_AET, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('Using Transformer_type: \'{}\' as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                        attention_type=cfg.MODEL.Attention_type,
                                                        sem=cfg.MODEL.SEM,
                                                        cem=cfg.MODEL.CEM,
                                                        sem_w=cfg.MODEL.SEM_W,
                                                        cem_w=cfg.MODEL.CEM_W,
                                                        ags=cfg.MODEL.AGS)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print(f'Loading ViT Pre-trained ImageNet model from {model_path}')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        self.AGS = cfg.MODEL.AGS
        self.att_feat_weight = cfg.MODEL.ATT_TRI_WEIGHT
        if self.AGS and cfg.MODEL.CEM:
            print(f"Use AGS! Mixed Feature Triplet Weight: {cfg.MODEL.ATT_TRI_WEIGHT}")
            self.att_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.att_classifier.apply(weights_init_classifier)
            self.att_bottleneck = nn.BatchNorm1d(self.in_planes)
            self.att_bottleneck.bias.requires_grad_(False)
            self.att_bottleneck.apply(weights_init_kaiming)
        else:
            cfg.defrost()
            cfg.MODEL.AGS = False
            cfg.freeze()
            self.AGS = False

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)


    def forward(self, x, label=None, cam_label=None, view_label=None):
        if self.AGS:
            global_feat, att_global_feat = self.base(x)
            att_feat = self.att_bottleneck(att_global_feat)
        else:
            global_feat = self.base(x)
        feat = self.bottleneck(global_feat)
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                if self.AGS:
                    att_cls_score = self.att_classifier(att_feat)
                    cls_score = self.classifier(feat)
                    return (cls_score, att_cls_score), (global_feat, att_global_feat)
                else:
                    cls_score = self.classifier(feat)
                    return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                if self.AGS:
                    return global_feat * (1 - self.att_feat_weight) + att_global_feat * self.att_feat_weight
                else:
                    return global_feat


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
