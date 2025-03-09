import torch
import utils
import os
from tqdm import tqdm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, config):
        super(DiffusiveRestoration, self).__init__()
        self.config = config
        self.diffusion = diffusion

        # 判断预训练模型是否存在
        pretrained_model_path = self.config.training.resume + '.pth.tar'
        assert os.path.isfile(pretrained_model_path), ('pretrained diffusion model path is wrong!')
        self.diffusion.load_ddm_ckpt(pretrained_model_path, ema=True)
        self.diffusion.model.eval()
        self.diffusion.model.requires_grad_(False)

    def restore(self, val_loader, r=None):
        image_folder = self.config.data.test_save_dir
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(val_loader)):
                print(f"=> starting processing image named {y}", flush=True)
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond, r=r)
                x_output = inverse_data_transform(x_output)
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))

    def diffusive_restoration(self, x_cond, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list

    def web_restore(self, image, r=None):
        with torch.no_grad():
            image_cond = image.to(self.diffusion.device)
            image_output = self.diffusive_restoration(image_cond, r=r)
            image_output = inverse_data_transform(image_output)
            return image_output
