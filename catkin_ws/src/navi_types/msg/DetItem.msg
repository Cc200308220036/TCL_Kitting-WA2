# ros message for one detect result

string label   # Object label
float32 confidence   # Detection confidence
int16[4] bbox   # Detection bounding box, [x1, y1, x2, y2]
sensor_msgs/Image mask   # Mask image  
