## 4batch_yolov5s

#### 1. 简介
​		在本实验中，我们将大图像分割为四个大小相等的小图像，并经过预处理后以batch大小为4的形式输入YOLOv5s模型进行推理和后处理。这种方法有效避免了因非等比例缩放而导致的图像失真，同时减少了缩放操作带来的像素信息损失，避免了像素填充操作的需求。此外，我们使用官方预训练的YOLOv5模型，复现过程相对简单，为目标检测任务提供了高效且准确的解决方案。 

#### 2. 环境准备

请根据需求自行准备环境

```
import os
import cv2
import argparse
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
```

#### 3. 运行程序

程序中设置了多个参数，可以根据需求自行修改。

```
parser.add_argument('--detect_model',type=str, default=r'yolov5s_4b.onnx', help='model.pt path(s)') 
parser.add_argument('--batch_size', type=int, default=4, help='inference size (pixels)')
parser.add_argument('--display', type=bool, default=False, help='inference size (pixels)')
parser.add_argument('--output_path', type=str, default='out4', help='source') 
parser.add_argument('--image_path', type=str, default='imgs', help='source') 
parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf_thresh', type=float, default=0.6, help='source') 
parser.add_argument('--nms_thresh', type=float, default=0.6, help='source')
parser.add_argument('--h_iou', type=float, default=0.7, help='source')
```

**display**设置为**True**会展示预处理后的4张小图。
**batch_size**设置为**1**或**4**，与模型同步。
**image_path**是输入图像的文件夹路径。
**conf_thresh**和**nms_thresh**正常设置即可
**h_iou**是H方向上边的IOU，超过多少多少合并框