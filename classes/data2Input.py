import os, os.path
import numpy as np
import general.path as path
import general.dataType as TYPE
from general.util import createFolder, read_file, write_file


def positionToSeq2SeqInput(decoder_target_tokens, num_total_data, encoder_file_folder, decoder_file_folder):
    # decoder_target_tokens = {'{': 0, '}':1, 'row':2, 'col': 3, 'START': 4, 'EOS': 5}
    temp_encoder_all_data =  []
    temp_decoder_all_data =  []
    max_encoder_len = 0
    max_decoder_len = 0
    for i in range(num_total_data):
        positions = read_file(encoder_file_folder+str(i)+TYPE.TXT, 'splitlines')
        gui = read_file(decoder_file_folder+str(i)+TYPE.GUI, 'splitBySpec')
        # print(gui)
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
    decoder_input_data = np.zeros((num_total_data, max_decoder_len, len(decoder_target_tokens)), dtype='float32')
    for i, (temp_data, gui) in enumerate(zip(temp_encoder_all_data, temp_decoder_all_data)):
        for j, data in enumerate(temp_data):
            encoder_input_data[i, j] = data
        for j, token in enumerate(gui):
            decoder_input_data[i, j, decoder_target_tokens[token]] = 1
        decoder_input_data[i, j+1:, decoder_target_tokens['EOS']] = 1

    return encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len
        

    




if __name__ == "__main__":
    print(path.DATASET1_ROWCOL_POSITION_TXT)
    list1 = os.listdir(path.DATASET1_ROWCOL_POSITION_TXT)
    num_total_data = len(list1)
    
    encoder_input_data, decoder_input_data = positionToSeq2SeqInput(2, path.DATASET1_ROWCOL_POSITION_TXT, path.DATASET1_ROWCOL_GUI)
    print('encoder_input_data', encoder_input_data)
    print('decoder_input_data', decoder_input_data)