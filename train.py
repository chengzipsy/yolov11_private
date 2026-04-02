import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'yolo11l.yaml')
    #model.load('runs/train/gds2/weights/best.pt')  # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'dian.yaml',
                imgsz=160,
                epochs=100,
                batch=50,
                workers=0,
                exist_ok=False,
                device='',
                optimizer='auto',
                cos_lr=True,                #余弦退火
                close_mosaic=100,            #马赛克增强
                resume=False,               #断续训练
                project='runs/train',
                name='smartcar',
                single_cls=False,            #以训练位置为主要目的 而不是分类
                cache=True,                 #提高训练速度 增加内存使用率
                profile=False,               #针对onnx等模型进行优化
                )