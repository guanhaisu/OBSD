import cv2
import numpy as np
import os
import PIL.Image as Image
from tqdm import tqdm

def crop(image):
    # 获取图像的高度和宽度
    height, width, _ = image.shape
    # 指定要剪切的像素数量
    # pixels_to_remove = 5
    pixels_to_remove = 0

    # 剪切图像以去掉指定数量的像素
    image = image[pixels_to_remove:height - pixels_to_remove, pixels_to_remove:width - pixels_to_remove]
    # result = np.ones_like(image) * 255
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_value = 150
    ret, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 查找黑色像素的坐标
    black_pixels = np.column_stack(np.where(gray_image == 0))
    # 找到离上边框最近的黑色像素
    top_pixel = black_pixels[np.argmin(black_pixels[:, 0])]
    # 找到离下边框最近的黑色像素
    bottom_pixel = black_pixels[np.argmax(black_pixels[:, 0])]
    # 找到离左边框最近的黑色像素
    left_pixel = black_pixels[np.argmin(black_pixels[:, 1])]
    # 找到离右边框最近的黑色像素
    right_pixel = black_pixels[np.argmax(black_pixels[:, 1])]

    # 计算长方形的宽和高
    width = right_pixel[1] - left_pixel[1]
    height = bottom_pixel[0] - top_pixel[0]
    # 计算长方形的左上角坐标
    rectangle_top_left = (top_pixel[0], left_pixel[1])
    # 创建包含长方形内容的新图像
    rectangle_image = image[rectangle_top_left[0]:rectangle_top_left[0] + height,
                      rectangle_top_left[1]:rectangle_top_left[1] + width]
    # 保存新图像

    original_image = rectangle_image

    height, width, _ = original_image.shape

    # 计算正方形图片的边长，取较大的那个值
    side_length = max(width, height) + 6  # 增加6像素的边缘间隔

    # 创建一个白底的正方形图片
    square_image = np.ones((side_length, side_length, 3), dtype=np.uint8) * 255

    # 计算将原图放入正方形图片中的位置
    x_offset = (side_length - width) // 2
    y_offset = (side_length - height) // 2

    # 将原图粘贴到正方形图片的中间
    square_image[y_offset:y_offset + height, x_offset:x_offset + width] = original_image
    return square_image

# 读取"提取"下的所有图片
data_path = "your_data_path"
characters = os.listdir(data_path)
# 循环读取characters中的所有图片并使用crop进行处理后覆盖保存
for character in tqdm(characters):
    image = Image.open(os.path.join(data_path, character))
    # 将图片转换为numpy数组
    image = np.array(image)
    image = crop(image)
    # 转化为PIL图片
    image = Image.fromarray(image)
    # 保存图片
    image.save(os.path.join(data_path, character))
    # print(character + "处理完成")