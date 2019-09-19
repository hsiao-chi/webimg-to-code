from os import listdir
from os.path import isfile, join, splitext


def getFileNameList(dataPath, savedPath, savedName):
    files = [splitext(f)[0] for f in listdir(dataPath) if (isfile(join(dataPath, f)) and splitext(f)[-1] == '.gui')]
    with open(savedPath + savedName, 'w') as file:
        for f in files:
            file.write(f + '\n')


if __name__ == "__main__":
    # getFileNameList('E:\\projects\\webGener\\pix2code\\datasets\\pix2code_datasets\\web\\all_data\\', 'assest\\', 'filemane.txt')
    
    