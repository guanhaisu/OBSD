import argparse
import os
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from dataset import Data
from models import DenoisingDiffusion
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist




def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    parser = argparse.ArgumentParser()
    # 参数配置文件路径
    parser.add_argument("--config", default='configs.yml', type=str, required=False, help="Path to the config file")
    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()
    with open(os.path.join(args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    config = new_config
    config.local_rank = args.local_rank
    dist.init_process_group(backend='nccl')
    dist.barrier()
    world_size = dist.get_world_size()
    # 判断是否使用 cuda
    device = torch.device("cuda", config.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    print("=> using device: {}".format(device))
    config.device = device

    # 随机种子
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    # 加载数据
    DATASET = Data(config)
    _, val_loader = DATASET.get_loaders()

    # 创建模型
    print("=> creating denoising diffusion model")
    diffusion = DenoisingDiffusion(config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
