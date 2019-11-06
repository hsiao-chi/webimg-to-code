# import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import general.path as path
import general.dataType as TYPE
from general.util import createFolder, readFile, writeFile, write_file


def convert2RowCol(img, filter_last: bool=False):
    ret, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 20), np.uint8)
    convert = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    image, contours, hierarchy = cv2.findContours(
        convert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lastX, lastY, lastW, lastH = 0, 0, 0, 0
    lastIdx = 0
    detectionList = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if y > lastY:
            lastX, lastY, lastW, lastH = x, y, w, h
            lastIdx = i
        detectionList.append([0, x, y, w, h])
        cv2.rectangle(thresh, (x, y), (x+w, y+h), 255, -1)
    if filter_last:
        if lastIdx < len(detectionList):
            del detectionList[lastIdx]
        cv2.rectangle(thresh, (lastX, lastY), (lastX+lastW, lastY+lastH), 0, -1)
    detectionList.sort(key=lambda x: x[1])
    detectionList.sort(key=lambda x: x[2])
    # sortedList = sorted(detectionList, key=lambda l: l[1])
    # sortedList = sorted(sortedList, key=lambda l: l[1])

    return detectionList, thresh


def toYoloPosition(imgWidth, imgHigh, positions):
    yoloPosition = []
    for position in positions:
        element = position.split()
        yoloPosition.append([element[0], int(element[1])/imgWidth,
                             int(element[2])/imgHigh, int(element[3])/imgWidth, int(element[4])/imgHigh])
    return yoloPosition

def to_yolo_position_from_2dim_list(imgWidth, imgHigh, positions):
    yoloPosition = []
    for position in positions:
        print(position)
        yoloPosition.append([position[0], int(position[1])/imgWidth,
                             int(position[2])/imgHigh, int(position[3])/imgWidth, int(position[4])/imgHigh])
    return yoloPosition

def convert_to_position_and_rowcol_img(img_path, output_position_path, output_img_path):
    img = cv2.imread(img_path, 0)
    detectionList, rolColImg = convert2RowCol(img)
    yolo_position_list = to_yolo_position_from_2dim_list(img.shape[1], img.shape[0], detectionList)
    write_file(yolo_position_list, output_position_path, 2 )
    cv2.imwrite(output_img_path, rolColImg )
    return rolColImg, yolo_position_list



if __name__ == "__main__":
    dataFileNames = readFile(
        path.DATASET1_ASSEST, 'pix2code-dataset-filemane', TYPE.TXT, 'splitlines')
    for i, dataFileName in enumerate(dataFileNames):
        flag= True
        if flag:
            img = cv2.imread(path.PIX2CODE_ORIGIN_DATASET +
                             dataFileName + TYPE.IMG, 0)

            detectionList, rolColImg = convert2RowCol(img)
            print(len(detectionList), detectionList)
            writeFile(detectionList, path.DATASET1_ROWCOL_POSITION_TXT, str(i),  TYPE.TXT, 2 )

            # plt.subplot(1, 2, 1), plt.imshow(img, 'gray')
            # plt.subplot(1, 2, 2), plt.imshow(rolColImg, 'gray')
            # plt.show()
            # break
            cv2.imwrite(path.DATASET1_ROWCOL_PNG + str(i) + TYPE.IMG, rolColImg )

        else:
            img = cv2.imread(path.DATASET1_ROWCOL_PNG +str(i) + TYPE.IMG, 0)
            
            positions = readFile(path.DATASET1_ROWCOL_POSITION_TXT , str(i), TYPE.TXT, 'splitlines')
            yoloPositions = toYoloPosition(img.shape[0], img.shape[1], positions)
            writeFile(yoloPositions, path.DATASET1_ROWCOL_YOLO_POSITION_TXT, str(i), TYPE.TXT, 2 )


        print(i) if i % 100 == 0 else None

