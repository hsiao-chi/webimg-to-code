# import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from config import DataFileType, Path
import general


def convert2RowCol(img):
    ret, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 20), np.uint8)
    convert = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    image, contours, hierarchy = cv2.findContours(
        convert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(np.size(contours))
    lastX, lastY, lastW, lastH = 0, 0, 0, 0
    lastIdx = 0
    detectionList = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if y > lastY:
            lastX, lastY, lastW, lastH = x, y, w, h
            lastIdx = i
            # print(lastX, lastY)
        # print(x, y)
        detectionList.append([0, x, y, w, h])
        cv2.rectangle(thresh, (x, y), (x+w, y+h), 255, -1)
    # print(lastX, lastY)
    if lastIdx < len(detectionList):
        del detectionList[lastIdx]
    cv2.rectangle(thresh, (lastX, lastY), (lastX+lastW, lastY+lastH), 0, -1)
    sortedList = sorted(detectionList, key=lambda l: l[0])
    sortedList = sorted(sortedList, key=lambda l: l[1])

    return sortedList, thresh


if __name__ == "__main__":
    dataFileNames = general.readFile(
        Path.assest.value, 'pix2code-dataset-filemane', DataFileType.txt.value, 'splitlines')
    for i, dataFileName in enumerate(dataFileNames):
        img = cv2.imread(Path.originDataset.value +
                         dataFileName + DataFileType.img.value, 0)
        detectionList, rolColImg = convert2RowCol(img)
        print(len(detectionList), detectionList)
        general.writeFile(detectionList, Path.targetDataset.value, 'row-col-position-txt\\'+ str(i),  DataFileType.txt.value, 2 )
        # plt.subplot(1, 2, 1), plt.imshow(img, 'gray')
        # plt.subplot(1, 2, 2), plt.imshow(rolColImg, 'gray')
        # plt.show()
        # break
        cv2.imwrite(Path.targetDataset.value + 'row-col-png\\'+ str(i) + DataFileType.img.value, rolColImg )
