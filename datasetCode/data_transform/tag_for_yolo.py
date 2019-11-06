import msvcrt
import cv2
import numpy as np
from general.util import createFolder, read_file, write_file
import general.path as path
import general.dataType as TYPE
#  {button: 0, text: 1, title: 2}
def manual_class_tag(positions: list, origin_img):
    class_position = []
    (img_high, img_width, _) = origin_img.shape
    print(img_width, img_high)
    # cv2.imshow("origin_img", origin_img)
    for i, position in enumerate(positions):
        x, y = float(position[1])*img_width,float(position[2])*img_high
        w, h = float(position[3])*img_width, float(position[4])*img_high
        x, y, w, h = int(x), int(y), int(w), int(h)
        print(x, y, w, h, y+h)
        sub_img = origin_img[y :y+h,x : x+w]
        print(sub_img.shape)
        cv2.imshow("sub_img", sub_img)
        key = cv2.waitKey()
        class_position.append([str(key-48)]+position[1:])
        print(class_position[i])
        print(key)
        if i > 3:
            break
    return class_position

def manual_class_tag_from_file(img_path, poosition_path):
    read_positions = read_file(poosition_path, 'splitlines')
    positions = [position.split() for position in read_positions]
    img = cv2.imread(img_path)
    class_position = manual_class_tag(positions, img)
    return class_position
