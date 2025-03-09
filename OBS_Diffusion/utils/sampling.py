import torch
from torchvision.transforms.functional import crop
import tqdm as tqdm


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def generalized_steps(x, x_cond, seq, model, b, eta=0., device=None):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)

            et = model(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to(device))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(device))
    return xs, x0_preds


def generalized_steps_overlapping(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True,
                                  device=None, gen_diffusion=None):
    with torch.no_grad():
        if gen_diffusion is not None:
            b = torch.from_numpy(gen_diffusion.betas).float().to(device)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)

            at = compute_alpha(b, t.long())
            if torch.cuda.is_available() and device.index is None:
                num_gpus = torch.cuda.device_count()
                copied_t = [t.clone() for _ in range(num_gpus)]
                t = torch.cat(copied_t)
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)
            et_output = torch.zeros_like(x_cond, device=x.device)

            if manual_batching:
                manual_batching_size = 64
                xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners],
                                         dim=0)
                for k in range(0, len(corners), manual_batching_size):
                    if model.module.learn_sigma and gen_diffusion is not None:
                        c = x.shape[1]
                        model_output = model(torch.cat([x_cond_patch[k:k + manual_batching_size],
                                                   xt_patch[k:k + manual_batching_size]], dim=1), t)
                        outputs, model_var_values = torch.split(model_output, c, dim=1)
                    else:
                        outputs = model(torch.cat([x_cond_patch[k:k + manual_batching_size],
                                                   xt_patch[k:k + manual_batching_size]], dim=1), t)
                    for idx, (hi, wi) in enumerate(corners[k:k + manual_batching_size]):
                        et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
            else:
                for (hi, wi) in corners:
                    xt_patch = crop(xt, hi, wi, p_size, p_size)
                    x_cond_patch = crop(x_cond, hi, wi, p_size, p_size)
                    x_cond_patch = data_transform(x_cond_patch)
                    if model.module.learn_sigma and gen_diffusion is not None:
                        c = x.shape[1]
                        model_output = model(torch.cat([x_cond_patch, xt_patch], dim=1), t)
                        outputs, model_var_values = torch.split(model_output, c, dim=1)
                        et_output[:, :, hi:hi + p_size, wi:wi + p_size] += outputs
                    else:
                        et_output[:, :, hi:hi + p_size, wi:wi + p_size] += model(torch.cat([x_cond_patch, xt_patch],
                                                                                           dim=1), t)

            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to(device))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(device))

    return xs, x0_preds
