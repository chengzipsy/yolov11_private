import cv2
import os
import glob
import datetime
import time


def extract_frames_from_video(video_path, output_dir, frame_interval):
    """从单个视频文件中提取帧（带时间戳的唯一文件名）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return 0

    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 创建视频专用输出目录（带时间戳）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_output_dir = os.path.join(output_dir, f"{video_name}_frames_{timestamp}")
    os.makedirs(video_output_dir, exist_ok=True)

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n处理视频: {os.path.basename(video_path)}")
    print(f"总帧数: {total_frames}, FPS: {fps:.2f}, 帧间隔: {frame_interval}")

    saved_count = 0
    count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            # 生成带时间戳和唯一序列的文件名
            frame_timestamp = datetime.datetime.now().strftime("%H%M%S%f")[:-3]  # 毫秒级时间
            frame_name = f"{video_name}_{frame_timestamp}_{count:06d}.jpg"
            output_path = os.path.join(video_output_dir, frame_name)

            # 保存图片（90%质量以节省空间）
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_count += 1

            # 每10张打印一次进度
            if saved_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"已保存 {saved_count} 张图片, 耗时 {elapsed:.1f}秒...")

        count += 1

    cap.release()

    # 计算处理速度
    end_time = time.time()
    duration = end_time - start_time
    fps_rate = saved_count / duration if duration > 0 else 0

    print(f"完成! 共保存 {saved_count} 张图片")
    print(f"处理耗时: {duration:.1f}秒, 平均 {fps_rate:.1f} FPS")
    print(f"图片保存至: {os.path.abspath(video_output_dir)}")
    return saved_count


def process_folder(input_folder, output_base_dir, frame_interval):
    """处理文件夹中的所有MP4文件"""
    # 确保输出基础目录存在
    os.makedirs(output_base_dir, exist_ok=True)

    # 添加运行时间戳到基础目录
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_run_dir = os.path.join(output_base_dir, f"extraction_{run_timestamp}")
    os.makedirs(output_run_dir, exist_ok=True)

    # 查找所有MP4文件（包括子目录）
    video_files = glob.glob(os.path.join(input_folder, "**", "*.mp4"), recursive=True)
    video_files.extend(glob.glob(os.path.join(input_folder, "**", "*.MP4"), recursive=True))  # 大写扩展名

    if not video_files:
        print(f"在 {input_folder} 中未找到任何MP4文件")
        return

    print(f"在 {input_folder} 中找到 {len(video_files)} 个MP4文件")
    print(f"帧提取间隔: 每 {frame_interval} 帧提取一张")
    print(f"输出目录: {os.path.abspath(output_run_dir)}")

    total_images = 0
    start_time = time.time()

    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'=' * 50}")
        print(f"处理视频 {i}/{len(video_files)}: {os.path.basename(video_path)}")
        images_count = extract_frames_from_video(video_path, output_run_dir, frame_interval)
        total_images += images_count

    # 计算总处理时间
    end_time = time.time()
    total_duration = end_time - start_time

    print(f"\n{'=' * 50}")
    print(f"所有视频处理完成!")
    print(f"共处理 {len(video_files)} 个视频")
    print(f"总保存图片数: {total_images} 张")
    print(f"总耗时: {total_duration:.1f}秒")
    print(f"所有图片保存在: {os.path.abspath(output_run_dir)}")

    # 生成完成标记文件
    complete_file = os.path.join(output_run_dir, "COMPLETE.txt")
    with open(complete_file, "w") as f:
        f.write(f"视频提取任务完成于 {datetime.datetime.now()}\n")
        f.write(f"处理视频数: {len(video_files)}\n")
        f.write(f"总图片数: {total_images}\n")
        f.write(f"总耗时: {total_duration:.1f}秒\n")


if __name__ == "__main__":
    # 用户输入
    input_folder = 'datasets/video'
    output_base_dir = 'datasets/new_dian'
    frame_interval = 8

    # 执行处理
    process_folder(input_folder, output_base_dir, frame_interval)
    print("\n处理完成! 按Enter键退出...")