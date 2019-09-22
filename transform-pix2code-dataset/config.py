from enum import Enum

class DataFileType(Enum):
    gui = '.gui'
    img = '.png'
    txt = '.txt'

class Path(Enum): 
    originDataset = 'E:\\projects\\webGener\\pix2code\\datasets\\pix2code_datasets\\web\\all_data\\'
    targetDataset = 'E:\\projects\\NTUST\\webimg-to-code\\dataset\\pix2code\\'
    assest = 'E:\\projects\\NTUST\\webimg-to-code\\transform-pix2code-dataset\\assest\\'