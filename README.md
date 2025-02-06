<p align="center">
    <img src="https://s2.loli.net/2024/03/19/JnbZmeh18VsqkxF.png" width="250" style="margin-bottom: 0.2;"/>
<p>

<h2 align="center"> <a href="https://arxiv.org/abs/2406.00684">OBSD: Deciphering Oracle Bone Language with Diffusion Models </a></h2>
<h4 align="center">Haisu Guan, Huanxin Yang, Xinyu Wang, Shengwei Han, Yongge Liu, Lianwen Jin, Xiang Bai, Yuliang Liu </h4>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2406.00684-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.00684) 
[![Demo](https://img.shields.io/badge/Demo-blue)](http://vlrlab-monkey.xyz:7680/OBSD)
</h5>


## News 
* ```2024.8.14 ``` 🚀 OBSD has been selected as the ACL 2024 Best Paper.
* ```2024.7.15 ``` 🚀 OBSD has been selected as the ACL 2024 Oral.
* ```2024.5.16 ``` 🚀 OBSD is accepted by ACL 2024 Main. 
* ```2024.2.15 ``` 🚀 Sourced code for OBSD is released.

## Oracle Bone Script Decipher.

### Welcome to OBSD. [Paper](https://arxiv.org/abs/2406.00684) has been published, and we show the [Demo](http://27.18.157.241:7680/OBSD/).<br /> The dataset is available [here](https://github.com/RomanticGodVAN/character-Evolution-Dataset).<br /> 
In [data/modern_kanji.zip](./data/modern_kanji.zip), we also provide images of modern Chinese characters corresponding to Oracle Bone Script, and you can run the [data/process.py](./data/process.py) to process the data if you wish to use your own data.


## Data preparation

### You can arbitrarily divide the training and test sets from the dataset and place them in the following format. The image names in the input folder and the target folder need to correspond one to one. The input folder stores OBS images, and the target folder stores modern Chinese character images.
```plaintext
Your_dataroot/
├── train/  (training set)
│   ├── input/
│   │   ├── train_安_1.png (OBS image)
│   │   ├── train_安_2.png 
│   │   ├── train_北_1.png
│   │   └── train_北_2.png
│   └── target/
│       ├── train_安_1.png (Modern Chinese Character image)
│       ├── train_安_2.png 
│       ├── train_北_1.png 
│       └── train_北_2.png 
│
└── test/   (test set)
    ├── input/
    │   ├── test_1.png  (OBS image)
    │   └── test_2.png
    └── target/
        ├── test_1.png  (Modern Chinese Character image)
        └── test_2.png

```

### You also need to modify the following path to configs.yaml.
```yaml
data:
    train_data_dir: '/Your_dataroot/train/' # path to directory of train data
    test_data_dir: '/Your_dataroot/test/'   # path to directory of test data
    test_save_dir: 'Your_project_path/OBS_Diffusion/result' # path to directory of test output
    val_save_dir: 'Your_project_path/OBS_Diffusion/validation/'    # path to directory of validation during training
    tensorboard: 'Your_project_path/OBS_Diffusion/logs' # path to directory of training information

training:
    resume: '/Your_save_root/diffusion_model'  # path to pretrained model
```

## Train

### Environment Configuration
```bash
git clone https://github.com/guanhaisu/OBSD.git
cd OBS_Diffusion
```
```bash
conda create -n OBSD python=3.9
conda activate OBSD
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Start training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=1234 train_diffusion.py
```

### You can monitor the training process.
```bash
tensorboard --logdir ./logs
```

## Test
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=1234 eval_diffusion.py
```
### If you want to refine the generated character results, you can run the following script. Also be careful to change your file paths.
```bash
CUDA_VISIBLE_DEVICES=0 python refine.py
```
You can find the FontDiffuser weights here, [GoogleDrive](https://drive.google.com/drive/folders/1kRwi5sfHn6oufydDmd-7X9pPFDZzFjkk?usp=drive_link). 