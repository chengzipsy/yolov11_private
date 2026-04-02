import os
import json
import cv2

# 自定义类别映射
class_map = {
    0: 'object',  # 类别 0 对应的名称
}


# YOLO11的txt标签，转为Labelme的json文件
def yolo_to_labelme(txt_path, img_width, img_height):
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    shapes = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # 转换为绝对坐标
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # 计算矩形的四个顶点
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        shapes.append({
            'label': class_map[class_id],
            'points': [[x1, y1], [x2, y2]],
            'group_id': None,
            'shape_type': 'rectangle',
            'flags': {}
        })

    return shapes


def main():
    # 文件夹路径
    image_folder_path = r"datasets/smartcar_num_dataset/images"  # 这里指定图片的文件夹路径
    txt_folder_path = r"datasets/smartcar_num_dataset/labels"  # 这里指定YOLO的txt_labels的文件夹路径
    json_output_path = r"./datasets/seg-datasetsv2/json_labels"  # 这里指定待会生成Labelme的json_labels的文件夹路径

    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(json_output_path):
        os.makedirs(json_output_path)

    # 遍历所有txt文件并转换
    for txt_file in os.listdir(txt_folder_path):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(txt_folder_path, txt_file)

            # 获取与txt文件同名的图片路径
            img_file = txt_file.replace('.txt', '.jpg')  # 假设图片是jpg格式
            img_path = os.path.join(image_folder_path, img_file)

            try:
                # 使用OpenCV读取图片分辨率
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError(f"Image file not found: {img_path}")

                img_height, img_width, _ = img.shape

                shapes = yolo_to_labelme(txt_path, img_width, img_height)

                # 创建LabelMe格式的json文件
                labelme_data = {
                    'version': '4.5.6',
                    'flags': {},
                    'shapes': shapes,
                    'imagePath': img_file,
                    'imageData': None,
                    'imageHeight': img_height,
                    'imageWidth': img_width
                }

                json_path = os.path.join(json_output_path, txt_file.replace('.txt', '.json'))
                with open(json_path, 'w') as json_file:
                    json.dump(labelme_data, json_file, indent=2)

            except Exception as e:
                print(f"Error processing {img_file}: {e}")


# 主函数入口
if __name__ == "__main__":
    main()