from __future__ import print_function
from .config import *
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# input: [num_sample, max_input_seg_length, tokens]
# targets:[num_sample, max_target_seq_length, tokens]
def seq2seqTraining(encoder_input_data, decoder_input_data):
    
    num_sample, max_input_seq_length, num_input_token = encoder_input_data.shape
    num_sample, max_target_seq_length, num_target_token = decoder_input_data.shape
    
    decoder_target_data = np.roll(decoder_input_data, -1, axis=1)
    decoder_target_data[:,-1] = 0
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
    decoder_lstm = LSTM(LSTM_DECODER_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    decoder_dense = Dense(num_target_token, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=BATCH_SIZE,
            epochs=SEQ2SEQ_EPOCHES,
            validation_split=0.2)
    # Save model
    model.save(SEQ2SEQ_WEIGHT_SAVE_NAME)
