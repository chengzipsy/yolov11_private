import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'yolo11n.yaml')
    model.load('models/yolo11n.pt')  # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'coco8.yaml',
                imgsz=640,
                epochs=150,
                batch=25,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=45,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )