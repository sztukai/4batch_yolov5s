import os
import cv2
import argparse
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt

font_scale = 1.0  
font_thickness = 2  
font = cv2.FONT_HERSHEY_SIMPLEX  


CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop","mouse","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")


def allFilePath(rootPath,allFIleList):  #遍历文件
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def scale_coords(model_img_shape, coords, ori_img_shape):
    # Rescale coords (xyxy) from model_img_shape to ori_img_shape
    # calculate from ori_img_shape
    gain = min(model_img_shape[0] / ori_img_shape[0], model_img_shape[1] / ori_img_shape[1])  # gain  = old / new
    pad = (model_img_shape[1] - ori_img_shape[1] * gain) / 2, (model_img_shape[0] - ori_img_shape[0] * gain) / 2  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, ori_img_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2


def redirect_coords(coords, index, h, w):
    if index == 1:
        coords[:, [0,2]] += w
    if index == 2:
        coords[:, [1,3]] += h
    if index == 3:
        coords[:, [0,2]] += w
        coords[:, [1,3]] += h
    coords = coords[:, :4]
    return coords


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    # current shape [height, width]
    shape = im.shape[:2]

    # if new_shape = size  new_shape = (size, size)
    if isinstance(new_shape, int):            
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    # if resized:   r = 1
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # computing wh padding 
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    # divide padding into 2 sides
    dw /= 2  
    dh /= 2

    # resize
    if shape[::-1] != new_unpad:  
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, dw, dh


def img_preprocess1(img, MODEL_WIDTH=640, MODEL_HEIGHT=640):
    h0, w0 = img.shape[:2]
    r = 640 / max(h0, w0)

    # 1. 等比例缩放 或 等比放大
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

    # 2. padding
    img, dw, dh = letterbox(img, new_shape=(MODEL_HEIGHT, MODEL_WIDTH))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def img_preprocess2(img, MODEL_WIDTH=640, MODEL_HEIGHT=640):

    # 3. transpose
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)

    # 4. normolize
    img /= 255.0

    return img


def anchor_v3(image_post, label, lt_x, lt_y, rb_x, rb_y, score):
    
    label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    rect_width = label_size[0] + 10
    rect_height = label_size[1] + 10

    cv2.rectangle(image_post, (lt_x, lt_y), (rb_x, rb_y), (255, 0, 0), font_thickness)
    cv2.putText(image_post, label + ' ' + str(score), (lt_x, lt_y), font, font_scale, (0, 0, 255), font_thickness)
    print("==============anchors==============")
    return image_post


def NMS_1(boxes, iou_thresh):
    index = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while index.size > 0:
        i = index[0]
        keep.append(i)

        x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)

        inter_area = w * h
        area_a = (boxes[i, 2] - boxes[i, 0])*(boxes[i, 3] - boxes[i, 1])
        area_b = (boxes[index[1:], 2] - boxes[index[1:], 0])*(boxes[index[1:], 3] - boxes[index[1:], 1])

        iou = inter_area/(area_a + area_a - inter_area)

        idx = np.where(iou < iou_thresh)[0]

        index = index[idx + 1]
    return keep

def NMS_2(boxes, iou_thresh=0.5, h_iou=0.5):
    # 按类别划分。

    unique_groups = np.unique(boxes[:, 5])
    grouped = {group: boxes[boxes[:, 5] == group] for group in unique_groups}

    keep_boxes = []
    for cls, nmsed_boxes in grouped.items():

        # Non-Maximum Suppression (NMS)
        index = np.argsort(nmsed_boxes[:, 4])[::-1]  # Sort by score
        while index.size > 0:
            i = index[0]
            keep_boxes.append(nmsed_boxes[i])

            # Calculate IoU between the current box and the remaining ones
            x1 = np.maximum(nmsed_boxes[i, 0], nmsed_boxes[index[1:], 0])
            y1 = np.maximum(nmsed_boxes[i, 1], nmsed_boxes[index[1:], 1])
            x2 = np.minimum(nmsed_boxes[i, 2], nmsed_boxes[index[1:], 2])
            y2 = np.minimum(nmsed_boxes[i, 3], nmsed_boxes[index[1:], 3])

            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)

            inter_area = w * h
            area_a = (nmsed_boxes[i, 2] - nmsed_boxes[i, 0]) * (nmsed_boxes[i, 3] - nmsed_boxes[i, 1])
            area_b = (nmsed_boxes[index[1:], 2] - nmsed_boxes[index[1:], 0]) * (nmsed_boxes[index[1:], 3] - nmsed_boxes[index[1:], 1])

            # Calculate IoU
            iou0 = inter_area / (area_a + area_b - inter_area)
            iou1 = inter_area / area_a
            iou2 = inter_area / area_b

            # Conditions to keep the boxes
            condition1 = iou0 <= iou_thresh
            condition2 = iou1 <= iou_thresh
            condition3 = iou2 <= iou_thresh

            idx = np.where(condition1 & condition2 & condition3)[0]
            index = index[idx + 1]  # Update index

    return np.array(keep_boxes)

def merge_and_nms(boxes, iou_thresh=0.5, h_iou=0.5):
    # 按类别划分。

    unique_groups = np.unique(boxes[:, 5])
    grouped = {group: boxes[boxes[:, 5] == group] for group in unique_groups}

    keep_boxes = []
    for cls, nmsed_boxes in grouped.items():
        #   NMS_2
        index = np.argsort(nmsed_boxes[:, 4])[::-1]  # Sort by score
        temp_list = []
        while index.size > 0:
            i = index[0]
            temp_list.append(nmsed_boxes[i])

            # Calculate IoU between the current box and the remaining ones
            x1 = np.maximum(nmsed_boxes[i, 0], nmsed_boxes[index[1:], 0])
            y1 = np.maximum(nmsed_boxes[i, 1], nmsed_boxes[index[1:], 1])
            x2 = np.minimum(nmsed_boxes[i, 2], nmsed_boxes[index[1:], 2])
            y2 = np.minimum(nmsed_boxes[i, 3], nmsed_boxes[index[1:], 3])

            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)

            inter_area = w * h
            area_a = (nmsed_boxes[i, 2] - nmsed_boxes[i, 0]) * (nmsed_boxes[i, 3] - nmsed_boxes[i, 1])
            area_b = (nmsed_boxes[index[1:], 2] - nmsed_boxes[index[1:], 0]) * (nmsed_boxes[index[1:], 3] - nmsed_boxes[index[1:], 1])

            # Calculate IoU
            iou0 = inter_area / (area_a + area_b - inter_area)
            iou1 = inter_area / area_a
            iou2 = inter_area / area_b

            # Conditions to keep the boxes
            condition1 = iou0 <= iou_thresh
            condition2 = iou1 <= iou_thresh
            condition3 = iou2 <= iou_thresh

            idx = np.where(condition1 & condition2 & condition3)[0]
            index = index[idx + 1]  # Update index

        # merge and remove
        nmsed_boxes = np.array(temp_list)
        new_index = np.argsort(nmsed_boxes[:, 4])[::-1]
        Len = nmsed_boxes.shape[0]

        for i in range(Len):
            box1 = nmsed_boxes[new_index[i]]
            j = i + 1
            while j < Len:
                box2 = nmsed_boxes[new_index[j]]
                x1 = max(box1[0], box2[0], 0)
                y1 = max(box1[1], box2[1], 0)
                x2 = min(box1[2], box2[2], )
                y2 = min(box1[3], box2[3], )
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                inter = w * h

                Iou = inter / ((box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter)
                Iou1 = max(h / (box1[3] - box1[1]), h / (box2[3] - box2[1]))

                
                # 如果有交集，并且 H 方向的交并比大于iou_threash, 合并框
                if Iou > 0 and Iou1 > h_iou:
                    nmsed_boxes[i][0] = min(box1[0], box2[0])
                    nmsed_boxes[i][1] = min(box1[1], box2[1])
                    nmsed_boxes[i][2] = max(box1[2], box2[2])
                    nmsed_boxes[i][3] = max(box1[3], box2[3])

                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                    # 根据面积占比获取新的置信度
                    nmsed_boxes[i][4] = (area1 * box1[4] + area2 * box2[4]) / (area1 + area2)

                    # 置信度小的框设置为非法框
                    nmsed_boxes[j][0] = nmsed_boxes[j][1] = nmsed_boxes[j][2] = nmsed_boxes[j][3] = -100

                j += 1

        nmsed_boxes = nmsed_boxes[nmsed_boxes[:, 0] > 0]

        keep_boxes.extend(nmsed_boxes)

    return np.array(keep_boxes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model',type=str, default=r'yolov5s_4b.onnx', help='model.pt path(s)')  #检测模型
    parser.add_argument('--batch_size', type=int, default=4, help='inference size (pixels)')
    parser.add_argument('--display', type=bool, default=False, help='inference size (pixels)')
    parser.add_argument('--output_path', type=str, default='out4', help='source') 
    parser.add_argument('--image_path', type=str, default='imgs', help='source') 
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thresh', type=float, default=0.6, help='source') 
    parser.add_argument('--nms_thresh', type=float, default=0.6, help='source')
    parser.add_argument('--h_iou', type=float, default=0.7, help='source')
    opt = parser.parse_args()
    print('opt:',opt)

    # 获取文件夹下所有文件, 也就是待检测的图片
    file_list = []
    allFilePath(opt.image_path, file_list)

    # 加载检测模型
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    img_size = opt.img_size
    batch_size = opt.batch_size
    output_path = opt.output_path
    conf_thresh = opt.conf_thresh
    nms_thresh = opt.nms_thresh
    h_iou = opt.h_iou
    display = opt.display
    # 初始化推理环境
    session_detect = onnxruntime.InferenceSession(opt.detect_model, providers=providers)

    # 创建输出文件夹
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    count = 0
    for img_path in file_list:
        count += 1

        print(count, img_path)
        img = cv2.imread(img_path)
        IMG = img
        resize_img = img_preprocess1(img, img_size, img_size)
        if display == True:
            plt.imshow(resize_img)
            plt.show()
        resize_img = img_preprocess2(resize_img, img_size, img_size)
        
        # 获取分割的坐标点
        h, w = img.shape[:2]
        w1 = int(w*12/25)
        w2 = w - w1

        h2 = w2 
        h1 = h - h2
        if w2 > h:
            print('small w > big h')
            h1 = h2 = h

        # 根据坐标划分图片
        img0 = img[0:h2, 0:w2].copy()
        img1 = img[0:h2, w1:w].copy()
        img2 = img[h1:h, 0:w2].copy()
        img3 = img[h1:h, w1:w].copy()
        
        small_img_shape = img0.shape[:2] 

        # 图片预处理
        # 1. resize
        # 2. letterbox
        # 3. 640*640*3 -> 3*640*640
        # 4. normalize /=255.0
        img0 = img_preprocess1(img0, img_size, img_size)
        img1 = img_preprocess1(img1, img_size, img_size)
        img2 = img_preprocess1(img2, img_size, img_size)
        img3 = img_preprocess1(img3, img_size, img_size)

        if display == True:
            img_list = np.stack((img0,img1,img2,img3))

            # 检查图片是否分割正确
            for i in range(4):
                temp = img_list[i]
                title="img"+str(i)
                #行，列，索引
                plt.subplot(2, 2, i+1)
                plt.imshow(temp)
                plt.title(title)
            plt.show()


        img0 = img_preprocess2(img0, img_size, img_size)
        img1 = img_preprocess2(img1, img_size, img_size)
        img2 = img_preprocess2(img2, img_size, img_size)
        img3 = img_preprocess2(img3, img_size, img_size)

        print("img0:",img0.shape)
        print("img1:",img1.shape)
        print("img2:",img2.shape)
        print("img3:",img3.shape)

        # to 4*3*640*640
        img_list = np.stack((img0,img1,img2,img3))
               

        # # 如果改回 1 batch
        if batch_size == 1:
            small_img_shape = img.shape
            img_list = resize_img[np.newaxis,:]

        print('img_list:', img_list.shape) 
        infer_results = session_detect.run([session_detect.get_outputs()[0].name], {session_detect.get_inputs()[0].name: img_list})[0]
        print('y_onnx:', infer_results.shape)


        output_arr = []

        for i, output in enumerate(infer_results):

            choice = output[:, 4] > conf_thresh

            output = output[choice]
            print('output%d:'%(i) ,output.shape)

            classes_scores = output[:, 5:]
            classIds = np.argmax(classes_scores, axis=1)
            
            # 分数
            scores = output[:, 4]
            boxes = output[:, :4]

            print('classes_scores:', classes_scores.shape)
            print('classIds:', classIds.shape)
            print('scores:', scores.shape)
            print('boxes', boxes.shape)

            if boxes != []:
                boxes = xywh2xyxy(boxes)
                boxes = scale_coords((img_size, img_size), boxes, small_img_shape)
                boxes = redirect_coords(boxes, i, h1, w1)

            dets = np.concatenate((boxes, scores.reshape(-1, 1), classIds.reshape(-1, 1)), axis=1)

            indices = NMS_1(dets, nms_thresh)
            
            output_arr.append(dets[indices])

        outputs = np.concatenate(output_arr)
        print('outputs:', outputs.shape)
        if batch_size != 1:
            outputs = merge_and_nms(outputs, nms_thresh, h_iou)

            print('outputs:', outputs.shape)
        ori_img = IMG
        for result in outputs:
            score = result[4]
            result = [int(i) for i in result]
            left = result[0]
            top = result[1]
            right = result[2]
            bottom = result[3]
            score = int(score*100)
            classId = result[5]
            label = CLASSES[classId]

            ori_img = anchor_v3(IMG, label, left, top, right, bottom, score)
            print('label: %s, score: %s'%(label, score))
        ori_img, _, _ = letterbox(ori_img, (1080, 1920))
        cv2.imwrite('%s/img%d.jpg'%(output_path, count), ori_img)
            
            


