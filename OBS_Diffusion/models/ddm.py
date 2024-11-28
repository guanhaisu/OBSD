import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as tvu
import utils
import utils.misc as misc
from models.unet import DiffusionUNet
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import utils.lr_sched as lr_sched
from tqdm import tqdm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}
        self.back = {}

    def register(self, module):
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module, device):
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                # self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data.to(device)
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def store(self, module):
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.back[name] = param.data.clone()

    def restore(self, module):
        if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.back[name].data)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict, device):
        for name, param in state_dict.items():
            self.shadow[name] = param.to(device)

    def clear(self):
        self.back = {}


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, e, b):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float()).float()
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DenoisingDiffusion(object):
    def __init__(self, config, test=False):
        super().__init__()
        self.config = config
        self.device = config.device
        self.amp = config.training.amp
        self.writer = SummaryWriter(config.data.tensorboard)
        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.test = test
        if test:
            # if self.device.index is not None:
            #     self.model = torch.nn.DataParallel(self.model, device_ids=[self.device.index],
            #                                        output_device=self.device.index)
            # else:
            #     self.model = torch.nn.DataParallel(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[config.local_rank],
                                                                   output_device=config.local_rank)
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[config.local_rank],
                                                                   output_device=config.local_rank)
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        if config.training.amp:
            self.loss_scaler = NativeScaler()

        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, 'cpu')
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'], 'cuda')
        if self.amp:
            try:
                self.loss_scaler.load_state_dict(checkpoint['scaler'])
            except KeyError:
                print('=> failed to load scaler')
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        pretrained_model_path = self.config.training.resume + '.pth.tar'
        if os.path.isfile(pretrained_model_path):
            self.load_ddm_ckpt(pretrained_model_path)
        dist.barrier()

        # 训练
        for epoch in tqdm(range(self.start_epoch, self.config.training.n_epochs)):
            if dist.get_rank() == 0:
                print('=> current epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            train_loader.sampler.set_epoch(epoch)
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                lr_sched.adjust_learning_rate(self.optimizer, i / len(train_loader) + epoch, self.config.optim)

                # 混合精度训练
                self.optimizer.zero_grad()
                if self.amp:
                    with torch.cuda.amp.autocast():
                        loss = noise_estimation_loss(self.model, x, t, e, b)
                    self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(), update_grad=True)
                    self.optimizer.zero_grad()
                    torch.cuda.synchronize()
                    self.ema_helper.update(self.model, self.device)
                else:
                    loss = noise_estimation_loss(self.model, x, t, e, b)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    torch.cuda.synchronize()
                    self.ema_helper.update(self.model, self.device)

                current_lr = self.optimizer.param_groups[0]['lr']
                loss_value_reduce = misc.all_reduce_mean(loss.item())

                if self.step % 50 == 0 and dist.get_rank() == 0:
                    print(
                        'rank: %d, step: %d, loss: %.6f, lr: %.6f, time consumption: %.6f' % (
                            dist.get_rank(), self.step, loss_value_reduce, current_lr, data_time / (i + 1)))
                    self.writer.add_scalar('train/loss', loss_value_reduce, self.step)
                    self.writer.add_scalar('train/lr', current_lr, self.step)

                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.ema_helper.store(self.model)
                    self.ema_helper.ema(self.model)
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)
                    self.ema_helper.restore(self.model)
                    self.model.train()

            # self.scheduler.step()
            # 保存模型
            if (epoch % self.config.training.snapshot_freq == 0) and dist.get_rank() == 0:
                utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'step': self.step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema_helper': self.ema_helper.state_dict(),
                    'config': self.config,
                    'scaler': self.loss_scaler.state_dict() if self.amp else None
                }, filename=self.config.training.resume + '_' + str(epoch))
                utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'step': self.step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema_helper': self.ema_helper.state_dict(),
                    'config': self.config,
                    'scaler': self.loss_scaler.state_dict() if self.amp else None
                }, filename=self.config.training.resume)

        if dist.get_rank() == 0:
            utils.logging.save_checkpoint({
                'epoch': self.config.training.n_epochs,
                'step': self.step,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'ema_helper': self.ema_helper.state_dict(),
                'config': self.config,
                'scaler': self.loss_scaler.state_dict() if self.amp else None
            }, filename=self.config.training.resume + '_' + str(epoch))
            utils.logging.save_checkpoint({
                'epoch': self.config.training.n_epochs,
                'step': self.step,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'ema_helper': self.ema_helper.state_dict(),
                'config': self.config,
                'scaler': self.loss_scaler.state_dict() if self.amp else None
            }, filename=self.config.training.resume)

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.sampling.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size, device=self.device)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0., device=self.device)
        if last:
            xs = xs[0][-1]
        return xs

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.config.data.val_save_dir, str(self.config.data.image_size))
        with torch.no_grad():
            if dist.get_rank() == 0:
                print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)  # 条件图像
            x_gt = x[:, 3:, :, :].to(self.device)  # GT图像
            x_cond = data_transform(x_cond)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                # utils.logging.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                # utils.logging.save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))
                combined_image = torch.cat((x_cond[i].unsqueeze(0), x[i].unsqueeze(0), x_gt[i].unsqueeze(0)), dim=0)
                img = tvu.make_grid(combined_image, nrow=3, normalize=True, scale_each=True)
                utils.logging.save_image(img, os.path.join(image_folder, str(step), f"{i + 1}_{y[len(y) * i // n]}.png"))
