import argparse
import os
import yaml
import torch
import torch.utils.data
import numpy as np
from dataset import Data
from models import DenoisingDiffusion
from utils import misc
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
    parser.add_argument("--config", default='configs.yml', type=str, required=False,
                        help="Path to the config file")
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args()
    with open(os.path.join(args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    config = new_config

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
    misc.setup_for_distributed(args.rank == 0)

    # 判断是否使用 cuda
    config.local_rank = args.gpu
    device = torch.device("cuda", config.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    # print("=> using device: {}".format(device))
    config.device = device

    eff_batch_size = config.training.batch_size * config.training.patch_n * misc.get_world_size()
    assert config.optim.lr is not None
    config.optim.lr = config.optim.lr * eff_batch_size / 256

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
    if dist.get_rank() == 0:
        print("=> creating denoising diffusion model", flush=True)
        print("lr: %.2e" % config.optim.lr)
        print("effective batch size: %d" % eff_batch_size)
    diffusion = DenoisingDiffusion(config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
