from .TransReID import build_TransReID
from .CNN import Backbone
from .backbones.make_vit import vit_base_patch16_224, vit_small_patch16_224
from .AET import build_AET


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
}

def make_model(cfg, num_class, logger=None, **kwargs):
    if cfg.MODEL.NAME == 'AET-Net':
        model = build_AET(num_class, cfg, __factory_T_type)
        print('===========building AET-Net ===========')
    elif cfg.MODEL.NAME == 'transreid':
        camera_num = kwargs['camera_num']
        view_num = kwargs['view_num']
        model = build_TransReID(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
        print('===========building TransReID with JPM module ===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
