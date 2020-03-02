from nets.yolo3 import yolo_body
from keras.layers import Input

Inputs = Input([416, 416, 3])
model = yolo_body(Inputs, 3, 20)
model.summary()