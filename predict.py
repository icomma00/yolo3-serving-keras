from nets.yolo3 import yolo_body
from keras.layers import Input
from yolo import YOLO
from PIL import Image

def detect_img_for_test():
    yolo = YOLO()
    img_path = 'VOCdevkit/VOC2007/JPEGImages/000004.jpg'
    image = Image.open(img_path)
    r_image = yolo.detect_image(image)
    r_image.show()
    yolo.close_session()
    # r_image.save('xxx.png')

# 在终端手动输入图片路径进行识别
def detect_img_terminal():
    yolo = YOLO()
    while True:
        # 执行程序，在终端窗口上手动输入待预测图片文件路径
        img_path = 'VOCdevkit/VOC2007/JPEGImages/000004.jpg'
        img = input('Input image filename:')
        try:
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


if __name__ == '__main__':

    detect_img_terminal()
    # detect_img_for_test()