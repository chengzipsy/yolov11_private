import os
import cv2
import shutil
from tqdm import tqdm

# 配置路径
source_images_dir = "datasets/66/images"       # 原始图片目录
source_labels_dir = "datasets/66/labels"       # YOLO标注目录（.txt）
output_images_dir = "datasets/gds_resize/images"     # 输出图片目录
output_labels_dir = "datasets/gds_resize/labels"     # 输出标注目录
target_size = (480, 480)  # 宽, 高

# 创建输出目录
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# 支持的图像后缀
img_exts = ['.jpg', '.jpeg', '.png']

# 遍历图片
for filename in tqdm(os.listdir(source_images_dir)):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in img_exts:
        continue

    # 读取并缩放图像
    img_path = os.path.join(source_images_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图像: {img_path}")
        continue

    resized_img = cv2.resize(img, target_size)
    cv2.imwrite(os.path.join(output_images_dir, filename), resized_img)

    # 复制标注文件（同名 .txt）
    label_path = os.path.join(source_labels_dir, name + ".txt")
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(output_labels_dir, name + ".txt"))
    else:
        print(f"⚠️ 未找到对应标注: {label_path}")
