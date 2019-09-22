from os import listdir
from os.path import isfile, join, splitext
from enum import Enum
import cv2
from config import DataFileType, Path
import general


ROW = ['header', 'row', 'others']
COL = ['single', 'double', 'quadruple']
BRACKET = ['{', '}', ',']

def getFileNameList(dataPath, savedPath, savedName):
    files = [splitext(f)[0] for f in listdir(dataPath) if (isfile(join(dataPath, f)) and splitext(f)[-1] == DataFileType.gui)]
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

    dataFileNames = general.readFile(Path.assest.value, 'pix2code-dataset-filemane', DataFileType.txt.value, 'splitlines')
    for i, dataFileName in enumerate(dataFileNames):
        originList = general.readFile(Path.originDataset.value, dataFileName, DataFileType.gui.value, 'splitBySpec')
        targetList = toRowColData(originList)
        print(targetList)
        general.writeFile(targetList, Path.targetDataset.value, 'row-col-gui\\'+str(i), DataFileType.gui.value)
        img = cv2.imread(Path.originDataset.value + dataFileName + DataFileType.img.value)
        cv2.imwrite(Path.targetDataset.value + 'origin-png\\'+ str(i) + DataFileType.img.value, img )