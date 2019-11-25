from keras.models import Model
from keras.layers import Input, LSTM, Dense
import keras
import numpy as np
import tensorflow as tf
import os, os.path
import random
import general.path as path
import general.dataType as TYPE
from general.util import createFolder, readFile, writeFile, showLoss, showAccuracy
from classes.model.layoutGenerator import seq2seqTraining, SEQ2SEQ_EPOCHES
from classes.data2Input import positionToSeq2SeqInput


"""
decoder_token = dict(), eg. {'{': 0, '}':1, 'row':2, 'col': 3, 'EOS': 4}
"""
def seq2seq(input_seq, decoder_tokens, max_decoder_seq_length, encoder_model, decoder_model):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, len(decoder_tokens)))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, decoder_tokens['START']] = 1.

    reverse_decoder_tokens = dict(
        (i, token) for token, i in decoder_tokens.items())
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_decoder_tokens[sampled_token_index]
        if sampled_token != 'EOS':
            decoded_sentence.append(sampled_token)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_token == 'EOS' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, len(decoder_tokens)))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


if __name__ == "__main__":
    # SEQ2SEQ_EPOCHES = 2
    print(path.DATASET1_FULL_YOLO_POSITION_TXT)
    list1 = os.listdir(path.DATASET1_FULL_YOLO_POSITION_TXT)
    num_total_data = len(list1)
    
    createFolder(path.CLASS_SEQ2SEQ_PREDIT_GUI_PATH + str(SEQ2SEQ_EPOCHES))
    createFolder(path.CLASS_SEQ2SEQ_WEIGHT + str(SEQ2SEQ_EPOCHES))


    # decoder_target_tokens = {
    #     '{': 0, '}':1, 
    #     'row':2, 'header': 3, 'single': 4, 'double': 5, 'quadruple': 6,
    #     'title': 7, 'text': 8,
    #     'btn-active': 9, 'btn-inactive': 10, 'btn-green': 11, 'btn-orange': 12, 'btn-red': 13,   
    #     'START': 14, 'EOS': 15}

    decoder_target_tokens = {
        '{': 0, '}':1, 
        'row':2, 'col': 3,
        'title': 4, 'text': 5,
        'btn-active': 6, 'btn-inactive': 7, 'btn-green': 8, 'btn-orange': 9, 'btn-red': 10,   
        'START': 11, 'EOS': 12}
    
    encoder_input_data, decoder_input_data, decoder_tokens, max_decoder_len = positionToSeq2SeqInput(decoder_target_tokens, num_total_data, path.DATASET1_FULL_YOLO_POSITION_TXT, path.DATASET1_ROWCOL_ELEMENT_GUI)
    print('max_decoder_len:', max_decoder_len)
    encoder_model, decoder_model = seq2seqTraining(encoder_input_data, decoder_input_data, decoder_tokens)
    for i in range(10):
        ii = random.randint(0,num_total_data+1)
        input_seq = encoder_input_data[ii: ii+1]
        decoded_sentence = seq2seq(input_seq, decoder_tokens, max_decoder_len, encoder_model, decoder_model)
        print('decoded_sentence length: ', ii,len(decoded_sentence))
        writeFile(decoded_sentence, path.CLASS_SEQ2SEQ_PREDIT_GUI_PATH +str(SEQ2SEQ_EPOCHES) + '\\', str(ii), TYPE.GUI, dataDim = 1)
        