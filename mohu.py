import os
import shutil

import albumentations as A
import cv2
from tqdm import tqdm

# 输入文件夹（原图和标签）
input_images_dir = r"datasets/gds_resize/images"
input_labels_dir = r"datasets/gds_resize/labels"

# 输出文件夹（增强后的图和标签）
output_images_dir = r"datasets/gds_resize/images1"
output_labels_dir = r"datasets/gds_resize/labels1"

# 创建输出目录
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# 定义增强管道（保持参数不变）
transform = A.Compose(
    [
        A.Blur(blur_limit=(0, 6), p=0.5),  # 模糊
        # 你也可以加别的增强，比如：
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.5)
    ]
)

# 处理所有图片
for img_name in tqdm(os.listdir(input_images_dir), desc="Processing images"):
    img_path = os.path.join(input_images_dir, img_name)
    label_path = os.path.join(input_labels_dir, os.path.splitext(img_name)[0] + ".txt")

    # 读取图片
    image = cv2.imread(img_path)
    if image is None:
        print(f"跳过无法读取的图片: {img_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 应用增强
    transformed = transform(image=image)
    transformed_image = transformed["image"]

    # 保存增强后的图片（转回BGR存储）
    output_img_path = os.path.join(output_images_dir, img_name)
    cv2.imwrite(output_img_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

    # 复制标签文件
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(output_labels_dir, os.path.basename(label_path)))
    else:
        print(f"未找到标签文件: {label_path}")

print("✅ 图像和标签已处理完成")
