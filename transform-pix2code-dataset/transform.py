from os import listdir
from os.path import isfile, join, splitext
from enum import Enum
import cv2

class DataFileType(Enum):
    gui = '.gui'
    img = '.png'
    txt = '.txt'

class Path(Enum): 
    originDataset = 'E:\\projects\\webGener\\pix2code\\datasets\\pix2code_datasets\\web\\all_data\\'
    targetDataset = 'E:\\projects\\NTUST\\webimg-to-code\\dataset\\pix2code-row-col\\'
    assest = 'E:\\projects\\NTUST\\webimg-to-code\\transform-pix2code-dataset\\assest\\'

ROW = ['header', 'row', 'others']
COL = ['single', 'double', 'quadruple']
BRACKET = ['{', '}', ',']

def getFileNameList(dataPath, savedPath, savedName):
    files = [splitext(f)[0] for f in listdir(dataPath) if (isfile(join(dataPath, f)) and splitext(f)[-1] == DataFileType.gui)]
    with open(savedPath + savedName, 'w') as file:
        for f in files:
            file.write(f + '\n')

def readFile(filePath, fileName, fileType, spType = 'splitlines' or 'splitBySpec'):
    data = None
    with open(str(filePath) + fileName + str(fileType), 'r') as file:
        if spType == 'splitlines':
            data = file.read().splitlines()
        elif spType == 'splitBySpec':
            data = file.read().split()
    return data

def writeFile(data, filePath, fileName, fileType): 
    with open(str(filePath) + fileName + str(fileType), 'w+') as file:
        file.write(' '.join(data))

def toRowColData(originDataList):
    data = []
    headerFlag = False
    # commaFlag = False
    print(originDataList)
    for i, element in enumerate(originDataList):
        if element in BRACKET:
            data.append(element)
            if element == BRACKET[0]:
                # commaFlag = True
                pass
            else:
                headerFlag = False
                # commaFlag = False                
        elif element in COL:
            data.append('col')
        elif element == ROW[0]:
            data.append('row')
            headerFlag = True
        elif headerFlag:
            data.append('col')
        else:
            data.append('row')

        # if commaFlag and element not in BRACKET:
        #     data.append(',')
    return data

    


if __name__ == "__main__":
    
    
    # getFileNameList(originDataPath, 'assest\\', 'filemane.txt')

    dataFileNames = readFile(Path.assest.value, 'pix2code-dataset-filemane', DataFileType.txt.value, 'splitlines')
    for i, dataFileName in enumerate(dataFileNames):
        originList = readFile(Path.originDataset.value, dataFileName, DataFileType.gui.value, 'splitBySpec')
        targetList = toRowColData(originList)
        writeFile(targetList, Path.targetDataset.value, str(i), DataFileType.gui.value)
        img = cv2.imread(Path.originDataset.value + dataFileName + DataFileType.img.value)
        cv2.imwrite(Path.targetDataset.value + str(i) + DataFileType.img.value, img )
        break