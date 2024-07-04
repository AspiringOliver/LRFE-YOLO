import torch
from PIL import Image
import numpy as np

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='5_1y/weights/best.pt')

# 加载图像
img_path = '5_1y/25_1.jpg'
img = Image.open(img_path)

# 使用模型进行推理
results = model(img)

# 解析检测结果
detections = results.xywh[0].cpu().numpy()  # xywh格式 (x_center, y_center, width, height)

# 打印每个检测框的中心点坐标
for detection in detections:
    x_center, y_center, width, height, conf, cls = detection
    print(f'Class: {cls}, Confidence: {conf}, Center: ({x_center}, {y_center})')
