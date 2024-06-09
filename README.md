<p align="center">
    <img src="https://s2.loli.net/2024/03/19/JnbZmeh18VsqkxF.png" width="250" style="margin-bottom: 0.2;"/>
<p>

<h2 align="center"> <a href="https://arxiv.org/abs/2406.00684">OBSD: Deciphering Oracle Bone Language with Diffusion Models </a></h2>
<h4 align="center">Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, Xiang Bai </h4>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2311.06607-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.00684) 
</h5>


## News 
* ```2024.5.16 ``` 🚀 OBSD is accepted by ACL 2024 Main. 
* ```2024.2.15 ``` 🚀 Sourced code for OBSD is released.

## Oracle Bone Script Decipher.

### Welcome to OBSD. Paper is [here](https://arxiv.org/abs/2406.00684), and we show the [demo](http://vlrlab-monkey.xyz:7680/OBSD).<br /> The dataset is available [here](https://github.com/RomanticGodVAN/character-Evolution-Dataset).<br /> 
In [data/modern_kanji.zip](./data/modern_kanji.zip), we also provide images of modern Chinese characters corresponding to Oracle Bone Script, and you can run the [data/process.py](./data/process.py) to process the data if you wish to use your own data.


## Data preparation

### You can arbitrarily divide the training and test sets from the dataset and place them in the following format.
```plaintext
Your_dataroot/
├── train/  (training set)
│   ├── input/
│   │   ├── train_1.png (OBS image)
│   │   └── train_2.png
│   └── target/
│       ├── train_1.png (Modern Chinese Character image)
│       └── train_2.png 
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
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```


## Start training
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_diffusion.py
```

### You can monitor the training process.
```bash
tensorboard --logdir ./logs
```

## Test
```bash
CUDA_VISIBLE_DEVICES=0 python eval_diffusion.py
```
### If you want to refine the generated character results, you can run the following script. Also be careful to change your file paths.
```bash
CUDA_VISIBLE_DEVICES=0 python refine.py
```