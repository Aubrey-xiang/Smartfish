import argparse
import cv2.dnn
import numpy as np
from HIK import *
from send import *

# # coco80个类别
# CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 
# 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
# model_path = "yolov8n.onnx"
model_path = "fish1.onnx"
CLASSES = {0: 'die_fish',1: 'life_fish'}
# 80个类别对应80中随机颜色
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))#为每个类别分配一个 随机的 RGB 颜色

#鱼记数
def fish_num(death_fish, life_fish, class_id):
    if class_id == 0:
        death_fish = death_fish + 1
    if class_id == 1:
        life_fish = life_fish + 1
    return death_fish,life_fish



# 绘制
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    # 绘制矩形框
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    # 绘制类别
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(model_path):
    # 使用opencv读取onnx文件
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model_path)
    # 主动取流方式抓取图像
    cam, pData, nDataSize, stFrameInfo = open_camera()
    client = send()
    while True:
        death_fish = 0
        life_fish = 0
        frame = export_frame(cam, pData, nDataSize, stFrameInfo)
        original_image: np.ndarray = frame
        [height, width, _] = original_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        scale = length / 640 # 缩放比例
        # 设置模型输入
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        model.setInput(blob)
        # 推理
        outputs = model.forward() # output: 1 X 8400 x 84
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []
        # outputs有8400行，遍历每一行，筛选最优检测结果
        for i in range(rows):
            # 找到第i个候选目标在80个类别中，最可能的类别
            classes_scores = outputs[0][i][4:] # classes_scores:80 X 1
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    # cx cy w h  -> x y w h 
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box) #边界框
                scores.append(maxScore) # 置信度
                class_ids.append(maxClassIndex) # 类别
        # opencv版最极大值抑制
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                            round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
            # print("class_ids[index] = %d" % class_ids[index])
            death_fish, life_fish = fish_num(death_fish, life_fish, class_ids[index])
        image_show(original_image,"img")
        s = "life_fish= " + str(life_fish) + "   death_fish= " + str(death_fish)
        print(s)
        fish_count = death_fish + life_fish
        data = {
            "fish_count": fish_count,       #修改鱼的数目
            "fish_deaths": death_fish       #死鱼数目
        }
        client.publish(MQTT_PUB_TOPIC, "9" + json.dumps(data))
        time.sleep(2)

        
    



if __name__ == '__main__':
    main(model_path)

