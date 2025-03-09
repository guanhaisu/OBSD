import argparse
import os
import yaml
import torch
import numpy as np
from dataset import Data
from models import DenoisingDiffusion, DiffusiveRestoration
import torch.distributed as dist


def config_get():
    parser = argparse.ArgumentParser()
    # 参数配置文件路径
    parser.add_argument("--config", default='configs.yml', type=str, required=False, help="Path to the config file")
    args = parser.parse_args()

    with open(os.path.join(args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return new_config


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
    parser.add_argument('--local_rank', default=-1, type=int)
    config = config_get()
    args = parser.parse_args()

    # 判断是否使用 cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("=> using device: {}".format(device))
    config.device = device

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    args.dist_url = 'env://'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    # 判断是否使用 cuda
    config.local_rank = args.gpu
    device = torch.device("cuda", config.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    # print("=> using device: {}".format(device))
    config.device = device

    # 随机种子
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    # 加载数据
    DATASET = Data(config)
    val_loader = DATASET.get_loaders(parse_patches=False, test=True)

    # 创建模型
    print("=> creating diffusion model")
    diffusion = DenoisingDiffusion(config, test=True)
    model = DiffusiveRestoration(diffusion, config)

    # 恢复图像
    model.restore(val_loader, r=config.data.grid_r)


if __name__ == '__main__':
    main()
