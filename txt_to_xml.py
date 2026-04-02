import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document

# 配置路径
img_dir = 'datasets/smartcar_num_dataset/images'
label_dir = 'datasets/smartcar_num_dataset/labels'
save_dir = 'datasets/smartcar_num_dataset/xmls'
os.makedirs(save_dir, exist_ok=True)

# 类别名称列表（顺序要与训练时一致）
class_names = ['object']  # 修改为你自己的类别

def convert_yolo_to_voc(img_w, img_h, box):
    # YOLO: x_center, y_center, width, height (归一化)
    x_center, y_center, w, h = box
    xmin = int((x_center - w / 2) * img_w)
    ymin = int((y_center - h / 2) * img_h)
    xmax = int((x_center + w / 2) * img_w)
    ymax = int((y_center + h / 2) * img_h)
    return max(0, xmin), max(0, ymin), min(img_w - 1, xmax), min(img_h - 1, ymax)

def create_voc_xml(img_name, img_size, bboxes, save_path):
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode('images'))
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(img_name))
    annotation.appendChild(filename)

    size = doc.createElement('size')
    for tag, value in zip(['width', 'height', 'depth'], img_size):
        el = doc.createElement(tag)
        el.appendChild(doc.createTextNode(str(value)))
        size.appendChild(el)
    annotation.appendChild(size)

    for bbox in bboxes:
        class_id, xmin, ymin, xmax, ymax = bbox

        obj = doc.createElement('object')
        name = doc.createElement('name')
        name.appendChild(doc.createTextNode(class_names[class_id]))
        obj.appendChild(name)

        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        obj.appendChild(difficult)

        bndbox = doc.createElement('bndbox')
        for tag, val in zip(['xmin', 'ymin', 'xmax', 'ymax'], [xmin, ymin, xmax, ymax]):
            el = doc.createElement(tag)
            el.appendChild(doc.createTextNode(str(val)))
            bndbox.appendChild(el)

        obj.appendChild(bndbox)
        annotation.appendChild(obj)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(doc.toprettyxml(indent='  '))

# 遍历图像并转换标注
for img_name in os.listdir(img_dir):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
    if not os.path.exists(label_path):
        continue

    img = cv2.imread(image_path)
    h, w, c = img.shape

    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            xmin, ymin, xmax, ymax = convert_yolo_to_voc(w, h, bbox)
            boxes.append([class_id, xmin, ymin, xmax, ymax])

    xml_save_path = os.path.join(save_dir, os.path.splitext(img_name)[0] + '.xml')
    create_voc_xml(img_name, (w, h, c), boxes, xml_save_path)

print("✅ YOLO 标注已成功转换为 VOC XML 格式！")
