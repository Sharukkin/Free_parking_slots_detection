import argparse
import cv2
from flask import Flask, Response
from video_flow import ParkingDetector
import pickle5 as pickle

app = Flask(__name__)

parser = argparse.ArgumentParser(description='Free parking lots script')
parser.add_argument('--video', type=str, required=True,
                    help='path to video file', default=None)
parser.add_argument('--rtsp', type=str, required=False,
                    help='path to rtsp flow', default=None)

args = parser.parse_args()

video_file = args.video
rtsp = args.rtsp
input_source = video_file if video_file is not None else rtsp
rtsp_mode = False if rtsp is None else True

with open("regions.p", 'rb') as f:
    regions_of_parking_plates = pickle.load(f)
    dict_parked_car_places = dict()
    for id_parking_plate in range(len(regions_of_parking_plates)):
        dict_parked_car_places.update({id_parking_plate: regions_of_parking_plates[id_parking_plate]})


def gen(video_source):
    id_frame = 0
    while True:
        frame = video_source.get_frame(id_frame);
        id_frame += 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame.tostring() + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(ParkingDetector(input_source, rtsp_mode)), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)


