# CNN Attention Enhanced Transformer for Occluded Person Re-Identification

We propose AET-Net, a CNN Attention Enhanced Transformer Network for ReID to solve the occlusion problem. Achieving state-of-the-art performanceon on occluded dataset Occluded-Duke.

## News
- 2022.x  Release the code of AET-Net.

## Pipeline

![framework](https://github.com/Peitong-Li/AET-Net/blob/main/imgs/readme_framework.png)
![MAS](https://github.com/Peitong-Li/AET-Net/blob/main/imgs/readme_MAS.png)

## Comparison results between AET-Net and the state-of-the-art methods
![occ_duke](https://github.com/Peitong-Li/AET-Net/blob/main/imgs/OCC_Duke.png)
![market_duke](https://github.com/Peitong-Li/AET-Net/blob/main/imgs/M_D.png)


## Abaltion Study of AET-NET

![Ablation](https://github.com/Peitong-Li/AET-Net/blob/main/imgs/Ablation.png)
![Portable_ablation](https://github.com/Peitong-Li/AET-Net/blob/main/imgs/portable_ablation.png)
![Inferential Costs](https://github.com/Peitong-Li/AET-Net/blob/main/imgs/portable_ablation.png)



## Requirements
### Installation

```bash
conda create -n AET-NET python=3.8 -y
conda activate AET-NET
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
(we use /torch 1.8.1 /torchvision 0.9.1 /timm 0.5.4 /cuda 11.1 for training and evaluation.)
```

### Prepare Datasets

```bash
mkdir ../../datasets
```

Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775),[Occluded-Duke](https://github.com/lightas/Occluded-DukeMTMC-Dataset), 
then unzip them and rename them under the directory like

```
datasets
├── market1501
│   └── contents ..
├── MSMT17
│   └── contents ..
├── dukemtmcreid
│   └── contents ..
├── Occluded_Duke
│   └── contents ..
```

### Prepare ViT Pre-trained Models

You need to download the ImageNet pretrained transformer model : [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth), [ViT-Small](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth)

## Training

We utilize one 3090 GPU for training.

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --config_file configs/OCC_Duke/baseline.yml --tag ${TAG} MODEL.NAME ${0} MODEL.DEVICE_ID "('your device id')" MODEL.STRIDE_SIZE ${1} MODEL.Attention_type ${2} MODEL.SEM ${3} MODEL.SEM_W ${4} MODEL.SEM_P ${5}  MODEL.CEM ${6} MODEL.CEM_W ${7} MODEL.CEM_P ${8} MODEL.AGS ${9} OUTPUT_DIR ${OUTPUT_DIR} DATASETS.NAMES "('your dataset name')" 
```

#### Arguments
- `${tag}`: the tag of running, e.g. 'test' or 'train', default='default'
- `${0}`: the type of model to build, e.g. 'AET-Net' or 'TransReID'
- `${1}`: stride size for pure transformer, e.g. [16, 16], [14, 14], [12, 12]
- `${2}`: the ways of build attention module, e.g. 'RGA' or 'CBAM'
- `${3}`: whether using SEM, True or False.
- `${4}`: the weight of SEM: [0, 1].
- `${5}`: the location of the SEM, 'before', 'after', 'all' (default='before').
- `${6}`: whether using CEM, True or False.
- `${7}`: the weight of CEM: [0, 1].
- `${8}`: the location of the CEM, 'before', 'after' (default='after').
- `${9}`: whether usiong AGS, True or False
- `${OUTPUT_DIR}`: folder for saving logs and checkpoints, e.g. `baseline`, the result will output to './logs/{datasets}/{model.NAMES}/baseline/TAG'


**or you can directly train with following yml and commands:**

```bash
# OCC_DukeMTMC AET-NET baseline
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --config_file configs/OCC_Duke/baseline.yml --tag 'train' MODEL.NAME 'AET-Net' MODEL.DEVICE_ID "('0')"
# OCC_DukeMTM AET-NET (baseline + SEM(RGA))
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --config_file configs/OCC_Duke/AET-NET/RGA/SEM.yml --tag 'train' MODEL.NAME 'AET-Net' MODEL.DEVICE_ID "('0')" 
# OCC_DukeMTMC AET-NET (baseline + CEM(RGA))
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --config_file configs/OCC_Duke/AET-NET/RGA/CEM.yml --tag 'train' MODEL.NAME 'AET-Net' MODEL.DEVICE_ID "('0')"
# OCC_DukeMTMC AET-NET (baseline + SEM + CEM (RGA))
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --config_file configs/OCC_Duke/AET-NET/RGA/SC.yml --tag 'train' MODEL.NAME 'AET-Net' MODEL.DEVICE_ID "('0')"
# OCC_DukeMTMC TransReID (baseline + SEM + CEM + AGS (RGA))
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --config_file configs/OCC_Duke/AET-NET/RGA/SC_AGS.yml --tag 'train' MODEL.NAME 'AET-Net' MODEL.DEVICE_ID "('0')"

# DukeMTMC baseline
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --config_file configs/Duke/baseline.yml --tag 'train' MODEL.NAME 'AET-Net' MODEL.DEVICE_ID "('0')"
# Market baseline
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --config_file configs/Market/baseline.yml --tag 'train' MODEL.NAME 'AET-Net' MODEL.DEVICE_ID "('0')"
```

## Evaluation
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --config_file 'choose which config to test' --tag 'test' MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained checkpoints')" TEST.MAS (Whether to use MAS evaluation indicators) HEATMAP.SAVE (Whether to save the heat map) HEATMAP.ROOT "(Path to save the heat map)"
```

**Examples:**

```bash
# OCC_Duke
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --config_file configs/OCC_Duke/AET-NET/RGA/SC_AGS.yml --tag 'test' MODEL.DEVICE_ID "('0')" TEST.WEIGHT '../logs/occ_duke/AET-NET/RGA/SC_AGS/AET-NET_120.pth'
# Market
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --config_file configs/Market/AET-NET/RGA/SC_AGS.yml --tag 'test' MODEL.DEVICE_ID "('0')" TEST.WEIGHT '../logs/occ_duke/AET-NET/RGA/SC_AGS/AET-NET_120.pth'
```
## Visualization
![Visualization](https://github.com/Peitong-Li/AET-Net/blob/main/imgs/Visualization.jpg)
```bash
python Visualization.py --config_file "(you config path)" --use_cuda (Whether use cuda) --image_path "(The image path to be visualized)" --OUTPUT_DIR "(Path to Heat map output)" --Model_Type "(the tag of model)" --show (If show the image result) TEST.WEIGHT "(Path to your eval model)"
```
**Example:**

```bash
# OCC_Duke
python Visualization.py --config_file configs/OCC_Duke/AET-NET/RGA/SC_AGS.yml --use_cuda True --image_path ./demo/1.jpg  --OUTPUT_DIR ./demo/results --Model_Type "AET_SC_AGS" TEST.WEIGHT '../logs/occ_duke/AET-NET/RGA/SC_AGS/AET-NET_120.pth'
```

## Inference Costs
```bash
python calc_parms_and_flops.py --config_file "(you config path)" TEST.WEIGHT "(Path to your eval model)"
```

## Trained Models and logs

Note: The code will be released after the paper is accepted.

## Acknowledgement

Code base from [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) , [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [TransReID](https://github.com/damo-cv/TransReID), [vit-explain](https://github.com/jacobgil/vit-explain)


## Citation

## Contact
