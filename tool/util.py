import matplotlib.pyplot as plt
import os
import json
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

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
        if dataDim == 0:
            file.write(data)
        elif dataDim == 1:
            file.write(' '.join(data))
        elif dataDim == 2:
            for dataLine in data:
                file.write(' '.join(str(line) for line in dataLine) + '\n')
        elif dataDim == 'JSON':
            file.write(json.dumps(data))

def showLoss(history, save_path, title):
   
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(save_path + title + "-training-loss.png")
    plt.close()
    # plt.show()  

def showAccuracy(history, save_path, title):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(save_path + title + "-accuracy.png")
    plt.close()
    # plt.show()
