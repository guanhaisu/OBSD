#python sample.py \
#    --ckpt_dir="ckpt/" \
#    --content_image_path="data_examples/sampling/example_content.jpg" \
#    --style_image_path="/home/ipad_ocr/yhx/data/甲骨文手动去重新版最终版__抠图_H_3.0/㠱/L_㠱_8720.jpg" \
#    --save_image \
#    --save_image_dir="outputs/" \
#    --device="cuda:0" \
#    --algorithm_type="dpmsolver++" \
#    --guidance_type="classifier-free" \
#    --guidance_scale=7.5 \
#    --num_inference_steps=20 \
#    --method="multistep"

python sample_modified.py --ckpt_dir ckpt/ --content_image_path="" --style_image_path="/home/ipad_ocr/yhx/FontDiffuser/data_examples/sampling/example_kaiti.png" --save_image --save_image_dir="outputs/" --device cuda:0 --algorithm_type dpmsolver++ --guidance_type classifier-free --guidance_scale 7.5 --num_inference_steps 20 --method multistep