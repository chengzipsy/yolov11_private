import os
import random
import shutil


def sample_and_rename_dataset(source_dir, target_dir, sample_size=50):
    """从图像分类数据集中每个类别随机抽取指定数量的图片， 重命名后复制到新文件夹中。.
    """
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"已创建目标文件夹: {target_dir}")

    # 获取源目录下所有的类别文件夹
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    if not classes:
        print("错误：在源目录中没有找到类别文件夹！")
        return

    total_copied = 0

    for cls_name in classes:
        cls_dir = os.path.join(source_dir, cls_name)

        # 获取该类别下所有的文件（过滤掉可能存在的隐藏文件，如 .DS_Store）
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(valid_exts)]

        # 确定要抽取的实际数量
        actual_sample_size = min(sample_size, len(images))

        if len(images) < sample_size:
            print(f"⚠️ 提示: 类别 [{cls_name}] 只有 {len(images)} 张图片，不足 {sample_size} 张，已全部提取。")

        # 随机抽取图片
        selected_images = random.sample(images, actual_sample_size)

        # 复制并重命名
        for img_name in selected_images:
            src_path = os.path.join(cls_dir, img_name)

            # 构造新文件名：类别名_原文件名
            new_name = f"{cls_name}_{img_name}"
            dst_path = os.path.join(target_dir, new_name)

            # 执行复制操作 (copy2 会保留文件的创建时间等元数据)
            shutil.copy2(src_path, dst_path)
            total_copied += 1

        print(f"✅ 类别 [{cls_name}] 处理完毕，已复制 {actual_sample_size} 张。")

    print(f"\n🎉 任务完成！总共复制了 {total_copied} 张图片到 {target_dir}")


if __name__ == "__main__":
    # ================= 配置区域 =================

    # 你的原始数据集路径（包含各个类别文件夹的根目录）
    SOURCE_DATASET_PATH = r"datasets\photos"

    # 你想存放提取出图片的全新目标路径
    TARGET_DATASET_PATH = r"datasets\datasets_all\images"

    # 每个类别抽取的数量
    NUM_SAMPLES = 50

    # ============================================

    sample_and_rename_dataset(SOURCE_DATASET_PATH, TARGET_DATASET_PATH, NUM_SAMPLES)
