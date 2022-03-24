# Free_parking_slots_detection
Detecting and counting free parking slots based on video stream of parking.

Project is based on yolov5. When bounding box of a car is fulfilling more than 0.1 space of parking slot, parking slot is counted as occupied. 
Model should be at least 0.71 confident that car is a car. Threshold was tuned to consistently detect cars and not to detect humans with shopping carts. 
For the purpose of marking up parking slots I created my own mark up programm.
Here you can see the result.

![Result](data/parking.gif)
