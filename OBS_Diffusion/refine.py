from FontDiffuser.sample import arg_parse, sampling, load_fontdiffuer_pipeline
import PIL.Image as Image
import os
import torch
import numpy as np

Font_args = arg_parse()
Font_args.demo = True
Font_args.ckpt_dir = 'FontDiffuser/ckpt'
pipe = load_fontdiffuer_pipeline(args=Font_args)


def run_fontdiffuer(source_image,
                    reference_image,
                    sampling_step,
                    guidance_scale,
                    seed):
    Font_args.character_input = False if source_image is not None else True
    Font_args.sampling_step = sampling_step
    Font_args.guidance_scale = guidance_scale
    Font_args.seed = seed
    if reference_image is None:
        reference_image = Image.open('FontDiffuser/figures/ref_imgs/繁体_龙.jpg')
    out_image = sampling(
        args=Font_args,
        pipe=pipe,
        content_image=source_image,
        style_image=reference_image)
    desired_size = (112, 112)
    out_image = out_image.resize(desired_size)
    return out_image


# 读取refine_path下所有图片
refine_path = 'Your_project_path/OBS_Diffusion/result'
refine_result_path = 'Your_project_path/OBS_Diffusion/result_refined'
if not os.path.exists(refine_result_path):
    os.makedirs(refine_result_path)
refine_files = os.listdir(refine_path)
seed = 61
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
for file in refine_files:
    if file.endswith('.png'):
        source_image = Image.open(os.path.join(refine_path, file))
        out_image = run_fontdiffuer(source_image, None, 20, 7.5, seed)
        out_image.save(os.path.join(refine_result_path, file))
        print(f'{file} refined')
