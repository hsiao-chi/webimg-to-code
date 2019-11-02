from os import listdir
from os.path import isfile, join, splitext
from enum import Enum
import cv2

import general.path as path
import general.dataType as TYPE
from general.util import createFolder, readFile, writeFile

ROW = ['header', 'row', 'others']
COL = ['single', 'double', 'quadruple']
BRACKET = ['{', '}', ',']

def getFileNameList(dataPath, savedPath, savedName):
    files = [splitext(f)[0] for f in listdir(dataPath) if (isfile(join(dataPath, f)) and splitext(f)[-1] == TYPE.GUI)]
    with open(savedPath + savedName, 'w') as file:
        for f in files:
            file.write(f + '\n')


def toRowColData(originDataList):
    data = []
    headerFlag = False
    for i, element in enumerate(originDataList):
        if element in BRACKET:
            data.append(element)
            if element == BRACKET[0]:
                pass
            else:
                headerFlag = False           
        elif element in COL:
            data.append('col')
        elif element == ROW[0]:
            data.append('row')
            headerFlag = True
        elif headerFlag:
            data.append('col')
        else:
            data.append('row')
    return data

    


if __name__ == "__main__":
    
    
    # getFileNameList(originDataPath, 'assest\\', 'filemane.txt')

    dataFileNames = readFile(path.DATASET1_ASSEST, 'pix2code-dataset-filemane', TYPE.TXT, 'splitlines')
    for i, dataFileName in enumerate(dataFileNames):
        originList = readFile(path.PIX2CODE_ORIGIN_DATASET, dataFileName, TYPE.GUI, 'splitBySpec')
        targetList = toRowColData(originList)
        print(targetList)
        writeFile(targetList, path.DATASET1_ROWCOL_GUI, str(i), TYPE.GUI)
        img = cv2.imread(path.PIX2CODE_ORIGIN_DATASET + dataFileName + TYPE.IMG)
        cv2.imwrite(path.DATASET1_ORIGIN_PNG + str(i) + TYPE.IMG, img )