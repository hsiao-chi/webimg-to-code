from __future__ import print_function
import general.path as path
import general.dataType as TYPE
from general.util import createFolder, readFile, writeFile, write_file, showLoss, showAccuracy
from keras.models import Model
from keras.layers import Input, LSTM, Dense, GaussianNoise, Bidirectional, Concatenate
from keras import backend as K, callbacks
from keras.utils import plot_model

import numpy as np
import tensorflow as tf
K.tensorflow_backend._get_available_gpus()

LSTM_ENCODER_DIM = 256  # Latent dimensionality of the encoding space.
LSTM_DECODER_DIM = 256
BATCH_SIZE = 64  # Batch size for training.
SEQ2SEQ_EPOCHES = 300  # Number of epochs to train for.
MODE_SAVE_PERIOD = 100
NUM_SAMPLE = 10000  # Number of samples to train on.

# input: [num_sample, max_input_seg_length, tokens]
# targets:[num_sample, max_target_seq_length, tokens]
#


def seq2seq_predit_model_old(model: Model)-> Model:
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[3].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.summary()

    decoder_inputs = model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(LSTM_DECODER_DIM,), name='input_3')
    decoder_state_input_c = Input(shape=(LSTM_DECODER_DIM,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.layers[4]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = model.layers[5]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    decoder_model.summary()
    return encoder_model, decoder_model

def seq2seq_predit_model(model: Model, bidirectional_lstm=False)-> Model:
    encoder_inputs = model.get_layer('encoder_input').input  # input_1
    if bidirectional_lstm:
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = model.get_layer('encoder_lstm').output
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        _decoder_dim = LSTM_DECODER_DIM*2
    else:
        encoder_outputs, state_h, state_c = model.get_layer('encoder_lstm').output
        _decoder_dim = LSTM_DECODER_DIM
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.summary()

    decoder_inputs = model.get_layer('decoder_input').input   # input_2
    decoder_state_input_h = Input(shape=(_decoder_dim,), name='input_3')
    decoder_state_input_c = Input(shape=(_decoder_dim,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = model.get_layer('decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    decoder_model.summary()
    return encoder_model, decoder_model

def bidirection_seq2seq_training_model(num_input_token, num_target_token, gaussian_noise=1) -> Model:
    encoder_inputs = Input(shape=(None, num_input_token), name="encoder_input")
    encoder = Bidirectional(LSTM(LSTM_ENCODER_DIM, return_state=True), name="encoder_lstm")
    _input =  encoder_inputs
    if gaussian_noise != None:
        encoder_noice = GaussianNoise(1, name="gaussian_noise")(encoder_inputs)
        _input = encoder_noice
    
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(_input)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_target_token), name="decoder_input")
    decoder_lstm = LSTM(
            LSTM_DECODER_DIM*2, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_target_token, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

def normal_training_model(num_input_token, num_target_token, gaussian_noise=1) -> Model: 
    encoder_inputs = Input(shape=(None, num_input_token), name="encoder_input")
    _input = encoder_inputs
    if gaussian_noise != None:
        encoder_noice = GaussianNoise(1, name="gaussian_noise")(encoder_inputs)
        _input = encoder_noice
    encoder = LSTM(LSTM_ENCODER_DIM, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder(_input)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None, num_target_token), name="decoder_input")
    decoder_lstm = LSTM(
            LSTM_DECODER_DIM, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_target_token, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def seq2seq_train_model(num_input_token, num_target_token,
                        optimizer='rmsprop', loss='categorical_crossentropy',
                        weight_path=None, gaussian_noise=1, encoder_bidirectional_lstm=False):

    if encoder_bidirectional_lstm:
        model = bidirection_seq2seq_training_model(num_input_token, num_target_token, gaussian_noise)
    else:
        model =  normal_training_model(num_input_token, num_target_token, gaussian_noise)


    # # Define an input sequence and process it.
    
    # encoder = LSTM(LSTM_ENCODER_DIM, return_state=True, name="encoder_lstm")
   
    # encoder_inputs = Input(shape=(None, num_input_token), name="encoder_input")
    # if gaussian_noise != None:
    #     encoder_noice = GaussianNoise(1, name="gaussian_noise")(encoder_inputs)
    #     encoder_outputs, state_h, state_c = encoder(encoder_noice)
    # else:
    #     encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        
    # # We discard `encoder_outputs` and only keep the states.
    # encoder_states = [state_h, state_c]
    # # Set up the decoder, using `encoder_states` as initial state.
    # decoder_inputs = Input(shape=(None, num_target_token), name="decoder_input")
    # # We set up our decoder to return full output sequences,
    # # and to return internal states as well. We don't use the
    # # return states in the training model, but we will use them in inference.
    
    # decoder_lstm = LSTM(
    #         LSTM_DECODER_DIM, return_sequences=True, return_state=True, name="decoder_lstm")
    # decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
    #                                      initial_state=encoder_states)

    # decoder_dense = Dense(num_target_token, activation='softmax', name="decoder_dense")
    # decoder_outputs = decoder_dense(decoder_outputs)

    # # Define the model that will turn
    # # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    # model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    # draw model graph
    # plot_model(model, to_file=SEQ2SEQ_MODEL_GRAPH_FILE)
    # Run training
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['accuracy'])
    model.summary()
    if weight_path:
        model.load_weights(weight_path)
    return model



    

def seq2seq_training(train_model: Model, encoder_input_data, decoder_input_data, decoder_target_token,
                     checkpoint_folder, analysis_saved_folder, final_model_saved_path, initial_epoch=0):

    decoder_target_data = np.roll(decoder_input_data, -1, axis=1)
    decoder_target_data[:, -1] = 0
    decoder_target_data[:, -1, decoder_target_token['EOS']] = 1

    mc = callbacks.ModelCheckpoint(checkpoint_folder + str(SEQ2SEQ_EPOCHES) + 'seq2seq-weights{epoch:05d}.h5',
                                   save_weights_only=True, period=MODE_SAVE_PERIOD)
    history = train_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                              batch_size=BATCH_SIZE,
                              epochs=SEQ2SEQ_EPOCHES,
                              validation_split=0.2,
                              initial_epoch=initial_epoch,
                              callbacks=[mc])
    showLoss(history, analysis_saved_folder, 'loss' + str(SEQ2SEQ_EPOCHES))
    showAccuracy(history, analysis_saved_folder,
                 'accuracy' + str(SEQ2SEQ_EPOCHES))
    write_file(history.history, analysis_saved_folder +
               'history'+str(SEQ2SEQ_EPOCHES)+TYPE.TXT, 'JSON')
    train_model.save(final_model_saved_path)
    return train_model

def seq2seq_evaluate(model: Model, encoder_input_data, decoder_input_data, decoder_target_token):
    decoder_target_data = np.roll(decoder_input_data, -1, axis=1)
    decoder_target_data[:, -1] = 0
    decoder_target_data[:, -1, decoder_target_token['EOS']] = 1
    loss, acc = model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data)
    print("\nLoss: %.2f, Accuracy: %.3f%%" % (loss, acc*100))



def seq2seq_predit(encoder_model: Model, decoder_model: Model, input_seq, decoder_tokens, max_decoder_seq_length, result_saved_path):
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

    write_file(decoded_sentence, result_saved_path, dataDim=1)
    return decoded_sentence
