# Parking_slots_counter
Counting free parking slots based on video flow on parking.

Project is based on yolov5. When bounding box of a car is fulfilling more than 0.1 space of parking slot, parking slot is counted as taken. 
Model should be at least 0.71 confident that car is a car. Threshold was calibrated to consistently detect cars  and not to detect human beings with shopping carts. 
For the purpose of marking up parking slots I created my own mark up programm.
Here you can see the result.

![Result](parking.gif)
