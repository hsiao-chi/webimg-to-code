from __future__ import print_function
import general.path as path
import general.dataType as TYPE
from general.util import createFolder, readFile, writeFile, showLoss, showAccuracy
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras import backend as K, callbacks
from keras.utils import plot_model

import numpy as np
import tensorflow as tf
K.tensorflow_backend._get_available_gpus()

LSTM_ENCODER_DIM = 256 # Latent dimensionality of the encoding space.
LSTM_DECODER_DIM = 256
BATCH_SIZE = 64  # Batch size for training.
SEQ2SEQ_EPOCHES = 400  # Number of epochs to train for.
MODE_SAVE_PERIOD = 50
NUM_SAMPLE = 10000  # Number of samples to train on.

# input: [num_sample, max_input_seg_length, tokens]
# targets:[num_sample, max_target_seq_length, tokens]
# 
def seq2seqTraining(encoder_input_data, decoder_input_data, decoder_target_token):

    num_sample, max_input_seq_length, num_input_token = encoder_input_data.shape
    num_sample, max_target_seq_length, num_target_token = decoder_input_data.shape

    decoder_target_data = np.roll(decoder_input_data, -1, axis=1)
    decoder_target_data[:, -1] = 0
    decoder_target_data[:, -1, decoder_target_token['EOS']] = 1
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_input_token))
    encoder = LSTM(LSTM_ENCODER_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_target_token))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(
        LSTM_DECODER_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_target_token, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # draw model graph
    # plot_model(model, to_file=SEQ2SEQ_MODEL_GRAPH_FILE)
    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    mc = callbacks.ModelCheckpoint(path.CLASS_SEQ2SEQ_WEIGHT + str(SEQ2SEQ_EPOCHES) +'\\' + 'seq2seq-weights{epoch:05d}.h5', 
                                     save_weights_only=True, period=MODE_SAVE_PERIOD)
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=BATCH_SIZE,
              epochs=SEQ2SEQ_EPOCHES,
              validation_split=0.2,
              callbacks=[mc])
    showLoss(history, path.CLASS_SEQ2SEQ_ANALYSIS_PATH, 'row-col-position-'+ str(SEQ2SEQ_EPOCHES))
    showAccuracy(history, path.CLASS_SEQ2SEQ_ANALYSIS_PATH, 'row-col-position-'+ str(SEQ2SEQ_EPOCHES))
    writeFile(history.history, path.CLASS_SEQ2SEQ_ANALYSIS_PATH, 'history'+str(SEQ2SEQ_EPOCHES), TYPE.TXT, 'JSON')
    # Save model
    # model.save(SEQ2SEQ_WEIGHT_SAVE_NAME)
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(LSTM_DECODER_DIM,))
    decoder_state_input_c = Input(shape=(LSTM_DECODER_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return encoder_model, decoder_model
