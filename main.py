from model.layoutGenerator import  seq2seqTraining
from tool.data2Input import positionToSeq2SeqInput
from tool.config import *
from tool.util import writeFile, createFolder
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import keras
import numpy as np
import tensorflow as tf
import os, os.path

"""
decoder_token = dict(), eg. {'{': 0, '}':1, 'row':2, 'col': 3, 'EOS': 4}
"""
def seq2seq(input_seq, decoder_tokens, max_decoder_seq_length, encoder_model, decoder_model):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, len(decoder_tokens)))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, decoder_tokens['EOS']] = 1.

    reverse_decoder_tokens = dict(
        (i, token) for token, i in decoder_tokens.items())
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_decoder_tokens[sampled_token_index]
        decoded_sentence += sampled_token+' '

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
    print(PATH_PIX2CODE_DATASET+PIX2CODE_POSITION_FOLDER)
    list1 = os.listdir(PATH_PIX2CODE_DATASET+PIX2CODE_POSITION_FOLDER)
    num_total_data = len(list1)
    
    createFolder(SEQ2SEQ_PREDIT_GUI_SAVE_PATH + str(SEQ2SEQ_EPOCHES))
    createFolder(SEQ2SEQ_WEIGHT_SAVE_PATH + str(SEQ2SEQ_EPOCHES))
    
    encoder_input_data, decoder_input_data, decoder_tokens, max_decoder_len = positionToSeq2SeqInput(num_total_data, PATH_PIX2CODE_DATASET+PIX2CODE_POSITION_FOLDER, PATH_PIX2CODE_DATASET+PIX2CODE_GUI_FOLDER)
    encoder_model, decoder_model = seq2seqTraining(encoder_input_data, decoder_input_data, decoder_tokens)
    for i in range(100):
        input_seq = encoder_input_data[i: i+1]
        decoded_sentence = seq2seq(input_seq, decoder_tokens, max_decoder_len, encoder_model, decoder_model)
        if i % 20 == 0:
            writeFile(decoded_sentence, SEQ2SEQ_PREDIT_GUI_SAVE_PATH +str(SEQ2SEQ_EPOCHES) + '\\', str(i), TYPE_GUI, dataDim = 0)
        