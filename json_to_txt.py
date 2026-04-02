import os
import json
from PIL import Image

def convert_to_yolo_bbox(box, img_width, img_height):
    # JSON: x, y 是左上角；YOLO: 需要中心点
    x_center = (box['x'] + box['width'] / 2) / img_width
    y_center = (box['y'] + box['height'] / 2) / img_height
    width = box['width'] / img_width
    height = box['height'] / img_height
    return x_center, y_center, width, height

def json_to_yolo(input_json_folder, output_txt_folder, image_folder=None, class_map=None):
    os.makedirs(output_txt_folder, exist_ok=True)

    for filename in os.listdir(input_json_folder):
        if not filename.endswith('.json'):
            continue

        json_path = os.path.join(input_json_folder, filename)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            image_name = item['image']
            txt_name = os.path.splitext(image_name)[0] + '.txt'
            txt_path = os.path.join(output_txt_folder, txt_name)

            # 图像路径
            img_path = os.path.join(image_folder if image_folder else input_json_folder, image_name)
            if not os.path.exists(img_path):
                print(f"⚠️ Warning: image {img_path} not found, skipping.")
                continue

            img = Image.open(img_path)
            img_width, img_height = img.size

            with open(txt_path, 'w') as out_file:
                for ann in item['annotations']:
                    label = ann['label']
                    class_id = class_map[label] if class_map and label in class_map else 0
                    bbox = ann['coordinates']
                    x, y, w, h = convert_to_yolo_bbox(bbox, img_width, img_height)

                    # 检查值合法性
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"❌ Skipping invalid box in {txt_name}: {x=:.2f}, {y=:.2f}, {w=:.2f}, {h=:.2f}")
                        continue

                    out_file.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ Done! YOLO labels saved to: {output_txt_folder}")


# 示例用法
if __name__ == "__main__":
    json_input_folder = r"datasets/dian_yolo/json"
    yolo_output_folder = r"datasets/dian_yolo/labels"
    image_folder = r"datasets/dian_yolo/images"  # 如果图片不在 JSON 文件夹下
    class_mapping = {
        "target": 0  # 自定义类别映射
    }

    json_to_yolo(json_input_folder, yolo_output_folder, image_folder, class_mapping)
