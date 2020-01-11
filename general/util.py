import matplotlib.pyplot as plt
import os
import json
import shutil


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def readFile(filePath, fileName, fileType, spType='splitlines' or 'splitBySpec'):
    data = None
    with open(str(filePath) + fileName + str(fileType), 'r') as file:
        if spType == 'splitlines':
            data = file.read().splitlines()
        elif spType == 'splitBySpec':
            data = file.read().split()
    return data


def writeFile(data, filePath, fileName, fileType, dataDim=1):
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


def read_file(file_path, spType='noSplit' or 'splitlines' or 'splitBySpec'):
    data = None
    with open(file_path, 'r') as file:
        if spType == 'splitlines':
            data = file.read().splitlines()
        elif spType == 'splitBySpec':
            data = file.read().split()
        elif spType == 'noSplit':
            data = file.read()
    return data


def write_file(data, file_path, dataDim=1):
    directory = file_path.split('\\')
    with open(file_path, 'w+') as file:
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


def copy_files(origin_folder, origin_start, origin_end, origin_file_type,
               target_folder, target_start, target_file_type,
               clean_target_folder=True):
    createFolder(origin_folder)
    createFolder(target_folder)
    target_listdir = os.listdir(target_folder)
    if (clean_target_folder and len(target_listdir) > 0):
        filelist = [f for f in target_listdir if f.endswith(target_file_type)]
        for f in filelist:
            os.remove(os.path.join(target_folder, f))

    for i, file_index in enumerate(range(origin_start, origin_end+1)):
        shutil.copy(origin_folder+str(file_index)+origin_file_type, target_folder+str(target_start+i)+target_file_type)

def replace_file_value(origin_file_path, target_file_path, toBeReplaced, target):
    file_content = read_file(origin_file_path, 'noSplit')
    file_content = file_content.replace(toBeReplaced, target)
    write_file(file_content, target_file_path, 0)