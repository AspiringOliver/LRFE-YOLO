from ultralytics import YOLO

if __name__ == '__main__':
    # 加载 YOLOv8s 模型
    model = YOLO('yolov8s.yaml').load('yolov8s.pt')
    # 训练模型
    results = model.train(data='../LRFE-YOLO/data/5.yaml', name='5_LRFE_YOLOv8s')
    results = model.train(data='../LRFE-YOLO/data/5_1.yaml', name='5_1_LRFE_YOLOv8s')

