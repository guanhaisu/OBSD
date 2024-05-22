import random
import gradio as gr
from sample import (arg_parse, sampling, load_fontdiffuer_pipeline)
from PIL import Image
import tempfile
tempfile.tempdir = "/data/JGW/yhx/gradio/tmp/"

def run_fontdiffuer(source_image,
                    reference_image,
                    sampling_step,
                    guidance_scale):
    args.character_input = False if source_image is not None else True
    args.sampling_step = sampling_step
    args.guidance_scale = guidance_scale
    args.seed = random.randint(0, 10000)
    out_image = sampling(
        args=args,
        pipe=pipe,
        content_image=source_image,
        style_image=reference_image)
    desired_size = (112, 112)
    out_image = out_image.resize(desired_size)
    return out_image


if __name__ == '__main__':
    args = arg_parse()
    args.demo = True
    args.ckpt_dir = 'ckpt'

    # load fontdiffuer pipeline
    pipe = load_fontdiffuer_pipeline(args=args)

    with gr.Blocks(title="风格转换") as demo:
        gr.HTML("""
    <nav style="background-color: #f8f9fa;color: white;padding: 16px;">
        <ul style="list-style-type: none;margin: 0;padding: 0;display: flex;">
            <li style="list-style-type: none;margin-right: 20px;">
                <a style="color: rgba(0,0,0,.5);text-decoration: none;font-size: 20px;" href="../">Home</a>
            </li>
            <li style="list-style-type: none;margin-right: 20px;">
                <a style="color: rgba(0,0,0,.5);text-decoration: none;font-size: 20px;" href="../radicals">Radical</a>
            </li>
            <li style="list-style-type: none;margin-right: 20px;">
                <a style="color: rgba(0,0,0,.5);text-decoration: none;font-size: 20px;" href="../EV">Evolution</a>
            </li>
            <li style="list-style-type: none;margin-right: 20px;">
                <a style="color: rgba(0,0,0,.5);text-decoration: none;font-size: 20px;" href="#">Pictographic</a>
            </li>
            <li style="list-style-type: none;margin-right: 20px;">
                <a style="color: rgba(0,0,0,.5);text-decoration: none;font-size: 20px;" href="../index/search/">Search</a>
            </li>
            <li style="list-style-type: none;margin-right: 20px;">
                <a style="color: rgba(0,0,0,.5);text-decoration: none;font-size: 20px;" href="#">Recognition</a>
            </li>
            <li style="list-style-type: none;margin-right: 20px;">
                <a style="color: rgba(0,0,0,.9);text-decoration: none;font-size: 20px;" href="../font">Veroeros</a>
            </li>
        </ul>
    </nav>
    """)
        with gr.Row():
            gr.Markdown("##   欢迎来到甲骨文文字风格转换模拟器，你只需要上传一张你想要转换的甲骨文图片，并指定一种字体风格样式的图片"
                        "就可以得到一张由模型生成的对应风格的甲骨文文字图片！"
                        "输入的甲骨文保证是一个单字文字，不要有其他干扰，同时为了更好的效果建议您上传白底黑字的二值图图片")
        with gr.Row():
            with gr.Column(scale=1):
                source_image = gr.Image(width=720, height=500, label=' 输入（上传你的图片）', image_mode='RGB',
                                        type='pil')
            with gr.Column(scale=1):
                reference_image = gr.Image(width=480, height=400,
                                           label='参考风格（可以从最下方选择你想上传的风格字样，也可以自己上传）',
                                           image_mode='RGB', type='pil')
            with gr.Column(scale=1):
                fontdiffuer_output_image = gr.Image(width=720, height=500, label="输出图片", image_mode='RGB',
                                                    type='pil')
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    sampling_step = gr.Slider(20, 50, value=20, step=1,
                                              label="Sampling Step（默认为20,可选20-50）",
                                              info="The sampling step by FontDiffuser.")
                with gr.Row():
                    guidance_scale = gr.Slider(1, 12, value=7.5, step=0.5,
                                               label="Scale of Classifier-free Guidance",
                                               info="The scale used for classifier-free guidance sampling")

            with gr.Column(scale=0.5):
                FontDiffuser = gr.Button('开始转换')

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Example : Source Image")
                gr.Markdown("### ⬇️以下提供了一些可以直接使用的甲骨文图片"
                            "你可以直接将它们拖拽入‘输入’一栏使用")
                gr.Examples(
                    examples=['jgwdata/㐁/H_㐁_60BB6_11.png',
                              'jgwdata/㒸/G_㒸_乙7674(甲).png',
                              'jgwdata/㘝/H_㘝_604B4_0.png',
                              'jgwdata/㝪/H_㝪_60ECB_11.png',
                              'jgwdata/㭉/G_㭉_佚898合8714賓組.png',
                              ],
                    examples_per_page=20,
                    inputs=source_image
                )
            with gr.Column(scale=1):
                gr.Markdown("## Example : Reference Image")
                gr.Markdown("### ⬇️以下提供了一些可以直接使用的参考风格字样，依次为卡通、正楷、篆书、繁体、草书、行书、隶书和其他三种风格"
                            "你可以直接将它们拖拽入‘参考风格’一栏使用")
                gr.Examples(
                    examples=['figures/ref_imgs/卡通_字.png',
                              'figures/ref_imgs/正楷_文.jpg',
                              'figures/ref_imgs/篆书_大.png',
                              'figures/ref_imgs/繁体_龙.jpg',
                              'figures/ref_imgs/草书_独.png',
                              'figures/ref_imgs/行书_行.jpg',
                              'figures/ref_imgs/隶书_福.jpg',
                              'figures/ref_imgs/ref_豄.jpg',
                              'figures/ref_imgs/ref_欟.jpg',
                              'figures/ref_imgs/ref_媚.jpg',
                              ],
                    examples_per_page=20,
                    inputs=reference_image
                )
        FontDiffuser.click(
            fn=run_fontdiffuer,
            inputs=[source_image,
                    reference_image,
                    sampling_step,
                    guidance_scale],
            outputs=fontdiffuer_output_image)
    demo.launch(server_port=7861, share=True, root_path='/font')
