from ultralytics import YOLO

# 加载一个模型，路径为 YOLO 模型的 .pt 文件
model = YOLO(r"runs/train/new_dian/weights/best.pt")

# 导出模型，设置多种参数
model.export(
    format="onnx",  # 导出格式为 ONNX
    imgsz=(480, 480),  # 设置输入图像的尺寸
    keras=False,  # 不导出为 Keras 格式9
    optimize=False,  # 移动设备优化的参数，用于在导出为TorchScript 格式时进行模型优化
    half=False,  # FP16 量化
    int8=False,  # INT8 量化
    data='dian.yaml',
    dynamic=True,  # 动态输入尺寸
    simplify=True,  # 简化 ONNX 模型
    opset=None,  # 使用最新的 opset 版本
    workspace=4.0,  # 为 TensorRT 优化设置最大工作区大小（GiB）
    nms=False,  # 不添加 NMS（非极大值抑制）
    batch=32,  # 指定批处理大小
    device="0"  # 指定导出设备为CPU或GPU，对应参数为"cpu" , "0"
)