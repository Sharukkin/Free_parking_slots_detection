import cv2
import time
import numpy as np
import pickle5 as pickle
from time import ctime
from functions import get_cars_dets_for_video, compute_busy_plates_by_frame


def read_regions():
    with open("regions.p", 'rb') as f:
        plates = pickle.load(f)
        dict_parked_car_places = dict()
        for number in range(len(plates)):
            dict_parked_car_places.update({number: plates[number]})
    return plates, dict_parked_car_places


class ParkingDetector:
    def __init__(self, input_source, rtsp_mode=False):
        self.car_bboxes = self.detect_cars(input_source)
        self.stream, self.frame_size, self.fps = self.run_stream(input_source, rtsp_mode)
        self.plates, self.dict_plates = read_regions()
        print(self.stream)

    def get_frame(self, id_frame):
        # start_time = time.time() + .00001
        success, frame = self.stream.read()
        
        # print(time.time() - start_time)
        # fps = 1/(time.time() - start_time)
        # print(f"Time: {ctime(start_time)} FPS : {fps}")
        while success:
            # start_time = time.time()
            # self.frame = cv2.resize(self.frame, (int(self.frame.shape[1]//2.4), int(self.frame.shape[0]//2.4)))
            # fps = 1/(time.time() - start_time)
            # print(f"Time: {ctime(start_time)} FPS : {fps}")
            
            rendered_frame = self.render_frame(frame, id_frame)
            #frame = self.render_frame(frame, id_frame)
            print(id_frame)
            return cv2.imencode('.JPEG', rendered_frame)[1]

    def run_stream(self, input_source, rtsp_mode=False, rtsp_latency=20):
        if rtsp_mode:
            g_stream = f"rtspsrc location={input_source} latency={rtsp_latency} ! decodebin ! videoconvert ! appsink"
            stream = cv2.VideoCapture(g_stream, cv2.CAP_GSTREAMER)
        else:
            stream = cv2.VideoCapture(input_source)
        frame_size = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = stream.get(cv2.CAP_PROP_FPS)
        datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        return stream, frame_size, fps

    def detect_cars(self, input_source):
        self.car_bboxes = get_cars_dets_for_video(input_source)
        return self.car_bboxes

    def render_frame(self, frame, id_frame):
        free_space = 0
        overlay = frame.copy()
        for box in self.car_bboxes[id_frame]:
            cv2.rectangle(frame, box[:2].astype(int), box[2:].astype(int), (0, 0, 250), 2)
        id_busy_plates = compute_busy_plates_by_frame(self.car_bboxes[id_frame], self.plates)
        for id_plate in self.dict_plates.keys():
            if id_plate not in id_busy_plates:
                cv2.fillPoly(frame, [np.array(self.dict_plates[id_plate])], (120, 200, 132))
                free_space += 1
            else:
                cv2.fillPoly(frame, [np.array(self.dict_plates[id_plate])], (0, 0, 200))

        cv2.putText(frame, f"free_space: {free_space}",
                    (int(self.frame_size[0] * 0.05) + 5, int(self.frame_size[1] * 0.15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 250), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, frame, 1 - 0.6, 0, frame)
        return frame