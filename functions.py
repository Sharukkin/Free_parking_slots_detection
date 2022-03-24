import cv2
import math
import torch
import numpy as np

import shapely
from shapely.geometry import Polygon

colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)


def get_color(c, x, max_val):
    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r * 255)


def plot_boxes_cv2(img, boxes, class_names=None, color=None):
    img = np.copy(img)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, 'car', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    return img


def get_cars_dets_for_video(input_source, score_thr=0.71, img_size=1024): #score_thr is a hyperparameter 
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="weights/best.pt")
    model.conf = score_thr
    stream = cv2.VideoCapture(input_source)
    grabbed = True
    bboxes = []

    while grabbed:
        grabbed, frame_bgr = stream.read()
        if not grabbed:
            continue

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = model(frame, size=img_size)

        result = results.pandas().xyxy[0]
        result = result[result["confidence"] > score_thr].sort_values(["confidence"], ascending=False)
        result = np.array(result)[:, :4]
        bboxes.append(result)

    bboxes = np.array(bboxes)
    return bboxes


def polygon_from_bbox(bbox):
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    p1 = [x1, y1]
    p2 = [x2, y1]
    p3 = [x2, y2]
    p4 = [x1, y2]
    return np.array([p1, p2, p3, p4])


def center_points_from_bboxes(bboxes):
    center_points = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        center_points.append([(y1 + (y2 - y1)/2), (x1 + (x2 - x1)/2)])
    return center_points


def compute_busy_plates_by_frame(car_bboxes_by_frame, car_plates):
    bbox_car_plates = []
    for id_plate in range(len(car_plates)):
        some_poly = Polygon(car_plates[id_plate])
        x, y = some_poly.exterior.coords.xy
        #bbox_car_plate = [min(list(set(x))), min(list(set(y))), max(list(set(x))), max(list(set(y)))]
        bbox_car_plate = [min(list(set(x))), max(list(set(y))), max(list(set(x))), min(list(set(y)))]
        bbox_car_plates.append(bbox_car_plate)

    plates_center_points = center_points_from_bboxes(bbox_car_plates)
    car_center_points = center_points_from_bboxes(car_bboxes_by_frame)

    id_busy_plates = []
    for id_car in range(len(car_bboxes_by_frame)):

        dist_between_car_and_plates = np.linalg.norm(
            np.array(car_center_points[id_car]) - np.array(plates_center_points), axis=1)
        top_5_dist = np.sort(dist_between_car_and_plates)[:5]
        ids_nearest_plates = [idx for idx, dis in enumerate(dist_between_car_and_plates) if dis in top_5_dist]

        polygon_car = Polygon(polygon_from_bbox(car_bboxes_by_frame[id_car]))

        IOU_dict = {}
        for id_plate in ids_nearest_plates:
            polygon_plate = Polygon(car_plates[id_plate])
            IOU = polygon_plate.intersection(polygon_car).area / polygon_plate.union(polygon_car).area
            IOU_dict[id_plate] = IOU
        key_by_max_value = max(IOU_dict, key=IOU_dict.get)
        if IOU_dict[key_by_max_value] >= 0.1:
            id_busy_plates.append(key_by_max_value)

    return id_busy_plates


