import os
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import PIL.Image
import re
import random
from torch.utils.data.distributed import DistributedSampler


class Data:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, test=False):

        train_path = os.path.join(self.config.data.train_data_dir)
        val_path = os.path.join(self.config.data.test_data_dir)

        train_dataset = MyDataset(train_path,
                                  patch_n=self.config.training.patch_n,
                                  patch_size=self.config.data.image_size,
                                  transforms=self.transforms,
                                  grid_r=self.config.data.grid_r,
                                  keep_image_size=self.config.data.training_keep_image_size,
                                  parse_patches=parse_patches)
        val_dataset = MyDataset(val_path,
                                patch_n=self.config.training.patch_n,
                                patch_size=self.config.data.image_size,
                                keep_image_size=self.config.data.testing_keep_image_size,
                                transforms=self.transforms,
                                grid_r=self.config.data.grid_r,
                                parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        # 训练数据

        # 评估数据
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
        #                                          shuffle=True, num_workers=self.config.data.num_workers,
        #                                          pin_memory=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True, sampler=DistributedSampler(val_dataset))

        if not test:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                       shuffle=False, sampler=DistributedSampler(train_dataset),
                                                       num_workers=self.config.data.num_workers,
                                                       prefetch_factor=2,
                                                       pin_memory=True)
            return train_loader, val_loader

        if test:
            return val_loader


# 数据集加载类
class MyDataset(torch.utils.data.Dataset):
    parse_patches: bool

    def __init__(self, dir_path, patch_size, patch_n, transforms, grid_r, keep_image_size, parse_patches=True):
        super().__init__()

        self.dir = dir_path
        input_names = os.listdir(dir_path + 'input')
        gt_names = os.listdir(dir_path + 'target')

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = patch_n
        self.grid_r = int(grid_r)
        self.keep_image_size = keep_image_size
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return [0], [0], h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = PIL.Image.open(os.path.join(self.dir, 'input', input_name)).convert(
            'RGB') if self.dir else PIL.Image.open(input_name)
        gt_img = PIL.Image.open(os.path.join(self.dir, 'target', gt_name)).convert(
            'RGB') if self.dir else PIL.Image.open(gt_name)

        if not self.keep_image_size:
            input_img = input_img.resize((100, 100), PIL.Image.Resampling.BILINEAR)
            gt_img = gt_img.resize((100, 100), PIL.Image.Resampling.BILINEAR)
        else:
            wd_new, ht_new = input_img.size
            wd_new = int(self.grid_r * np.ceil(wd_new / float(self.grid_r)))
            ht_new = int(self.grid_r * np.ceil(ht_new / float(self.grid_r)))
            assert wd_new >= self.patch_size and ht_new >= self.patch_size
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.Resampling.BILINEAR)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.Resampling.BILINEAR)

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id

        else:
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(np.ceil((wd_new - self.patch_size) / float(self.grid_r)) * self.grid_r + self.patch_size)
            ht_new = int(np.ceil((ht_new - self.patch_size) / float(self.grid_r)) * self.grid_r + self.patch_size)
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.Resampling.BILINEAR)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.Resampling.BILINEAR)
            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
