import os

import cv2

# ================= 配置区域 =================
# 你现有的图像分类数据集路径（里面是各个类别的文件夹）
SOURCE_DIR = r"runs\detect\photo_cropped_1"

# 修剪后图片存放的新路径（会自动按原样创建类别文件夹）
OUTPUT_DIR = r"runs\detect\photo_cropped_2"

# ✂️ 核心参数：上下左右各剪掉多少个像素？
# 比如设为 5，就会把图片左边切掉5像素，右边切掉5像素，上下同理
TRIM_MARGIN = 7
# ============================================


def trim_image_edges():
    total_processed = 0
    skipped = 0

    # 遍历源目录下的所有文件和子文件夹
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            # 过滤出图片文件
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                img_path = os.path.join(root, file)

                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ 无法读取图片: {img_path}")
                    continue

                h, w = img.shape[:2]

                # 安全检查：防止图片本来就太小，切完就没了
                if h <= 2 * TRIM_MARGIN or w <= 2 * TRIM_MARGIN:
                    print(f"⚠️ 图片太小，跳过裁切: {file} (尺寸: {w}x{h})")
                    skipped += 1
                    continue

                # ⭐️ 核心魔法：直接对 Numpy 数组进行切片，去掉四周边缘 ⭐️
                # img[起始Y:结束Y, 起始X:结束X]
                trimmed_img = img[TRIM_MARGIN : h - TRIM_MARGIN, TRIM_MARGIN : w - TRIM_MARGIN]

                # 构建输出路径 (保持原有的类别文件夹结构)
                rel_path = os.path.relpath(root, SOURCE_DIR)
                out_root = os.path.join(OUTPUT_DIR, rel_path)
                os.makedirs(out_root, exist_ok=True)

                save_path = os.path.join(out_root, file)

                # 保存图片 (默认保持原文件名)
                cv2.imwrite(save_path, trimmed_img)
                total_processed += 1

    print("\n🎉 批量修剪完成！")
    print(f"✅ 成功修剪并保存了 {total_processed} 张图片。")
    if skipped > 0:
        print(f"⚠️ 有 {skipped} 张图片因为尺寸太小而跳过。")
    print(f"📁 新图片已保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    trim_image_edges()
