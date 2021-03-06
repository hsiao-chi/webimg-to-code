from keras.models import Model
from keras.layers import Input, LSTM, Dense
import keras
import numpy as np
import tensorflow as tf
import os
import os.path
import random
import general.path as path
import general.dataType as TYPE
from general.util import createFolder, readFile, writeFile, showLoss, showAccuracy
from classes.model.layoutGenerator import seq2seqTraining, SEQ2SEQ_EPOCHES
from classes.data2Input import to_Seq2Seq_input
from general.node.nodeEnum import Font_color, Bg_color


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


def get_decoder_config(target_type=1):
    if target_type == 1:
        return {
            'data_folder': path.DATASET1_ORIGIN_GUI,
            'token_list': [
                '{', '}',
                'row', 'header', 'single', 'double', 'quadruple',
                'title', 'text',
                'btn-active', 'btn-inactive', 'btn-green', 'btn-orange', 'btn-red',
                'START', 'EOS']}
    elif target_type == 2:
        return {
            'data_folder': path.DATASET1_ROWCOL_ELEMENT_GUI,
            'token_list': [
                '{', '}',
                'row', 'col',
                'title', 'text',
                'btn-active', 'btn-inactive', 'btn-green', 'btn-orange', 'btn-red',
                'START', 'EOS']}
    elif target_type == 3:
        return {
            'data_folder': path.DATASET1_ROWCOL_ATTRIBUTE_GUI,
            'token_list': [
                '{', '}', '[', ']',
                'row', 'col',
                'title', 'text', 'btn',
                'text-white', 'text-primary', 'text-dark',
                'bg-primary', 'bg-dark', 'bg-success', 'bg-warning', 'bg-danger',
                'START', 'EOS']}


def get_encoder_config(target_type=1):
    if target_type == 1:
        return {
            'direct_part': 5,
            'data_folder': path.DATASET1_FULL_YOLO_POSITION_TXT,
            'class_mode': False,
            'token_list': [],
        }
    elif target_type == 2:
        return {
            'direct_part': 5,
            'data_folder': path.DATASET1_ATTRIBUTE_YOLO_POSITION_TXT,
            'class_mode': False,
            'token_list': [Font_color.dark.value, Font_color.primary.value, Font_color.white.value,
                           Bg_color.primary.value, Bg_color.dark.value, Bg_color.success.value,
                           Bg_color.warning.value, Bg_color.danger.value],
        }
    elif target_type == 3:
        return {
            'direct_part': 5,
            'data_folder': path.DATASET1_ATTRIBUTE_YOLO_POSITION_TXT,
            'class_mode': True,
            'token_list': [
                [Font_color.dark.value, Font_color.primary.value,
                    Font_color.white.value],
                [Bg_color.primary.value, Bg_color.dark.value, Bg_color.success.value,
                 Bg_color.warning.value, Bg_color.danger.value]
            ],
        }


if __name__ == "__main__":

    INPUT_TYPE = 3
    TARGET_TYPE = 1

    encoder_config = get_encoder_config(INPUT_TYPE)
    decoder_config = get_decoder_config(TARGET_TYPE)

    print(encoder_config['data_folder'])
    list1 = os.listdir(encoder_config['data_folder'])
    num_total_data = len(list1)

    createFolder(path.CLASS_SEQ2SEQ_PREDIT_GUI_PATH + str(SEQ2SEQ_EPOCHES))
    createFolder(path.CLASS_SEQ2SEQ_WEIGHT + str(SEQ2SEQ_EPOCHES))

    encoder_input_data, decoder_input_data, decoder_tokens, max_decoder_len = to_Seq2Seq_input(
        encoder_config['data_folder'], decoder_config['data_folder'], encoder_config, decoder_config['token_list'])
    print('---------encoder_input_data---------\n', encoder_input_data.shape)
    print('---------decoder_input_data---------\n', decoder_input_data.shape)

    print('max_decoder_len:', max_decoder_len)
    encoder_model, decoder_model = seq2seqTraining(
        encoder_input_data, decoder_input_data, decoder_tokens)
    for i in range(10):
        ii = random.randint(0, num_total_data+1)
        input_seq = encoder_input_data[ii: ii+1]
        decoded_sentence = seq2seq(
            input_seq, decoder_tokens, max_decoder_len, encoder_model, decoder_model)
        print('decoded_sentence length: ', ii, len(decoded_sentence))
        writeFile(decoded_sentence, path.CLASS_SEQ2SEQ_PREDIT_GUI_PATH +
                  str(SEQ2SEQ_EPOCHES) + '\\', str(ii), TYPE.GUI, dataDim=1)
