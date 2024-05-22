# OBSD


## Oracle Bone Script Decipher.

### Welcome to OBSD. We show the [demo](http://vlrlab-monkey.xyz:7680/OBSD) here.<br /> The dataset is available [here](https://github.com/RomanticGodVAN/character-Evolution-Dataset).<br /> 
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