from .util import readFile, writeFile
from .config import *
import os, os.path
import numpy as np


def positionToSeq2SeqInput(num_total_data, encoder_file_folder, decoder_file_folder):
    decoder_target_token = {'{': 0, '}':1, 'row':2, 'col': 3, 'START': 4, 'EOS': 5}
    temp_encoder_all_data =  []
    temp_decoder_all_data =  []
    max_encoder_len = 0
    max_decoder_len = 0
    for i in range(num_total_data):
        positions = readFile(encoder_file_folder, str(i), TYPE_TXT, 'splitlines')
        gui = readFile(decoder_file_folder, str(i), TYPE_GUI, 'splitBySpec')
        temp_data = []
        for position in  positions:
            p = position.split()
            temp_data.append(p)
        temp_encoder_all_data.append(temp_data)
        if len(temp_data) > max_encoder_len:
            max_encoder_len = len(temp_data)

        temp_decoder_all_data.append(['START']+gui)
        if len(gui)+1 > max_decoder_len:
            max_decoder_len = len(gui)+1
        


    encoder_input_data = np.zeros((num_total_data, max_encoder_len, 5), dtype='float32')
    decoder_input_data = np.zeros((num_total_data, max_decoder_len, len(decoder_target_token)), dtype='float32')
    for i, (temp_data, gui) in enumerate(zip(temp_encoder_all_data, temp_decoder_all_data)):
        for j, data in enumerate(temp_data):
            encoder_input_data[i, j] = data
        for j, token in enumerate(gui):
            decoder_input_data[i, j, decoder_target_token[token]] = 1
        decoder_input_data[i, j+1:, decoder_target_token['EOS']] = 1

    return encoder_input_data, decoder_input_data, decoder_target_token, max_decoder_len
        

    




if __name__ == "__main__":
    print(PATH_PIX2CODE_DATASET+PIX2CODE_POSITION_FOLDER)
    list1 = os.listdir(PATH_PIX2CODE_DATASET+PIX2CODE_POSITION_FOLDER)
    num_total_data = len(list1)
    
    encoder_input_data, decoder_input_data = positionToSeq2SeqInput(2, PATH_PIX2CODE_DATASET+PIX2CODE_POSITION_FOLDER, PATH_PIX2CODE_DATASET+PIX2CODE_GUI_FOLDER)
    print('encoder_input_data', encoder_input_data)
    print('decoder_input_data', decoder_input_data)