def readFile(filePath, fileName, fileType, spType = 'splitlines' or 'splitBySpec'):
    data = None
    with open(str(filePath) + fileName + str(fileType), 'r') as file:
        if spType == 'splitlines':
            data = file.read().splitlines()
        elif spType == 'splitBySpec':
            data = file.read().split()
    return data

def writeFile(data, filePath, fileName, fileType, dataDim = 1): 
    with open(str(filePath) + fileName + str(fileType), 'w+') as file:
        if dataDim == 1:
            file.write(' '.join(data))
        elif dataDim == 2:
            for dataLine in data:
                file.write(' '.join(str(line) for line in dataLine) + '\n')