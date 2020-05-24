from __future__ import print_function
import general.path as path
import general.dataType as TYPE
from general.util import createFolder, readFile, writeFile, write_file, showLoss, showAccuracy
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, GaussianNoise, Bidirectional, concatenate, Concatenate, Activation, dot, TimeDistributed, Embedding
from keras import backend as K, callbacks
from keras import activations
from keras.utils import plot_model
from enum import Enum
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

class SeqModelType(Enum):
    normal = 'normal'
    normal_attention = 'normal_attention'
    encoder_bidirectional = 'encoder_bidirectional'
    bidirectional = 'bidirectional'
    encoder_bidirectional_attention = 'encoder_bidirectional_attention'

def attention_section(encoder_outputs, decoder_outputs):
    attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2], name= "dot1")
    attention = Activation('softmax', name='attention')(attention)
    print('attention', attention)
    context = dot([attention, encoder_outputs], axes=[2,1], name= "context_output")
    print('context', context)
    return context

def normal_predit_model(model: Model, encoder_inputs, decoder_inputs):
    dh_input = Input(shape=(LSTM_DECODER_DIM,), name='input_h')
    dc_input = Input(shape=(LSTM_DECODER_DIM,), name='input_c')
    decoder_states_inputs = [dh_input, dc_input]
    _, eh, ec = model.get_layer('encoder_lstm').output
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, dh, dc = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    encoder_states = [eh, ec]
    decoder_states = [dh, dc]
    return encoder_states, decoder_states, decoder_outputs, decoder_states_inputs

def normal_attention_predit_model(model: Model, encoder_inputs, decoder_inputs):
    dh_input = Input(shape=(LSTM_DECODER_DIM,), name='input_h')
    dc_input = Input(shape=(LSTM_DECODER_DIM,), name='input_c')
    encoder_each_h_input = Input(shape=(None, LSTM_DECODER_DIM), name='input_each_h')

    decoder_states_inputs = [dh_input, dc_input]
    encoder_outputs, eh, ec = model.get_layer('encoder_lstm').output
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, dh, dc = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    context = attention_section(encoder_each_h_input, decoder_outputs)
    decoder_combined_context = concatenate([context, decoder_outputs])
    print('decoder_combined_context', decoder_combined_context)
    dense1 = model.get_layer('dense1')
    output = dense1(decoder_combined_context)
    
    encoder_states = [encoder_outputs, eh, ec]
    decoder_states = [dh, dc]
    return encoder_states, decoder_states, output, [encoder_each_h_input]+decoder_states_inputs

def normal_stack_predit_model(model: Model, encoder_inputs, decoder_inputs):
    dh0_input = Input(shape=(LSTM_DECODER_DIM,), name='input_h0')
    dc0_input = Input(shape=(LSTM_DECODER_DIM,), name='input_c0')
    decoder_state0 = [dh0_input, dc0_input]
    dh1_input = Input(shape=(LSTM_DECODER_DIM,), name='input_h1')
    dc1_input = Input(shape=(LSTM_DECODER_DIM,), name='input_c1')
    decoder_state1 = [dh1_input, dc1_input]
    decoder_state_inputs = [dh0_input, dc0_input, dh1_input, dc1_input]

    encoder_outputs, eh0, ec0 = model.get_layer('encoder_lstm_0').output
    encoder_lstm1 = model.get_layer('encoder_lstm')
    _, eh1, ec1 = encoder_lstm1(encoder_outputs)
    decoder_lstm0 = model.get_layer('decoder_lstm_0')
    decoder_outputs, dh0, dc0 = decoder_lstm0(decoder_inputs, initial_state=decoder_state0)
    decoder_lstm1 = model.get_layer('decoder_lstm')
    decoder_outputs, dh1, dc1 = decoder_lstm1(decoder_outputs, initial_state=decoder_state1)
    encoder_states = [eh0, ec0, eh1, ec1]
    decoder_states = [dh0, dc0, dh1, dc1]
    return encoder_states, decoder_states, decoder_outputs, decoder_state_inputs

def bidirectional_predit_model(model: Model, encoder_inputs, decoder_inputs):
    dfh_input = Input(shape=(LSTM_DECODER_DIM,), name='input_fh')
    dfc_input = Input(shape=(LSTM_DECODER_DIM,), name='input_fc')
    dbh_input = Input(shape=(LSTM_DECODER_DIM,), name='input_bh')
    dbc_input = Input(shape=(LSTM_DECODER_DIM,), name='input_bc')
    decoder_states_inputs = [dfh_input, dfc_input, dbh_input, dbc_input]
    
    _, efh, efc, ebh, ebc = model.get_layer('encoder_lstm').output
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, dfh, dfc, dbh, dbc = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    encoder_states = [efh, efc, ebh, ebc]
    decoder_states = [dfh, dfc, dbh, dbc]
    return encoder_states, decoder_states, decoder_outputs, decoder_states_inputs

def bidirectional_stack_predit_model(model: Model, encoder_inputs, decoder_inputs):
    dfh0_input = Input(shape=(LSTM_DECODER_DIM,), name='input_fh0')
    dfc0_input = Input(shape=(LSTM_DECODER_DIM,), name='input_fc0')
    dbh0_input = Input(shape=(LSTM_DECODER_DIM,), name='input_bh0')
    dbc0_input = Input(shape=(LSTM_DECODER_DIM,), name='input_bc0')
    decoder_states0 = [dfh0_input, dfc0_input, dbh0_input, dbc0_input]
    dfh1_input = Input(shape=(LSTM_DECODER_DIM,), name='input_fh1')
    dfc1_input = Input(shape=(LSTM_DECODER_DIM,), name='input_fc1')
    dbh1_input = Input(shape=(LSTM_DECODER_DIM,), name='input_bh1')
    dbc1_input = Input(shape=(LSTM_DECODER_DIM,), name='input_bc1')
    decoder_states1 = [dfh1_input, dfc1_input, dbh1_input, dbc1_input]
    decoder_states_inputs = [dfh0_input, dfc0_input, dbh0_input, dbc0_input, dfh1_input, dfc1_input, dbh1_input, dbc1_input]
    
    encoder_outputs, efh0, efc0, ebh0, ebc0 = model.get_layer('encoder_lstm_0').output
    encoder_lstm1 = model.get_layer('encoder_lstm')
    _, efh1, efc1, ebh1, ebc1 = encoder_lstm1(encoder_outputs)
    decoder_lstm0 = model.get_layer('decoder_lstm_0')
    decoder_outputs, dfh0, dfc0, dbh0, dbc0 = decoder_lstm0(decoder_inputs, initial_state=decoder_states0)
    decoder_lstm1 = model.get_layer('decoder_lstm')
    decoder_outputs, dfh1, dfc1, dbh1, dbc1 = decoder_lstm1(decoder_outputs, initial_state=decoder_states1)

    encoder_states = [efh0, efc0, ebh0, ebc0, efh1, efc1, ebh1, ebc1]
    decoder_states = [dfh0, dfc0, dbh0, dbc0, dfh1, dfc1, dbh1, dbc1]
    return encoder_states, decoder_states, decoder_outputs, decoder_states_inputs

def encoder_bidirectional_predit_model(model: Model, encoder_inputs, decoder_inputs):
    dh_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_h')
    dc_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_c')
    decoder_states_inputs = [dh_input, dc_input]

    _, efh, efc, ebh, ebc = model.get_layer('encoder_lstm').output
    state_h = Concatenate()([efh,  ebh])
    state_c = Concatenate()([efc, ebc])
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, dh, dc = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    encoder_states = [state_h, state_c]
    decoder_states = [dh, dc]
    return encoder_states, decoder_states, decoder_outputs, decoder_states_inputs

def encoder_bidirectional_attention_predit_model(model: Model, encoder_inputs, decoder_inputs):
    dh_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_h')
    dc_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_c')
    encoder_each_h_input = Input(shape=(None, LSTM_DECODER_DIM*2), name='input_each_h')
    decoder_states_inputs = [dh_input, dc_input]

    e_outputs, efh, efc, ebh, ebc = model.get_layer('encoder_lstm').output
    state_h = Concatenate()([efh,  ebh])
    state_c = Concatenate()([efc, ebc])
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, dh, dc = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    context = attention_section(encoder_each_h_input, decoder_outputs)
    decoder_combined_context = concatenate([context, decoder_outputs])
    print('decoder_combined_context', decoder_combined_context)
    dense1 = model.get_layer('dense1')
    output = dense1(decoder_combined_context)
    encoder_outputs = [e_outputs, state_h, state_c]
    decoder_states = [dh, dc]
    return encoder_outputs, decoder_states, output, [encoder_each_h_input]+decoder_states_inputs

def encoder_bidirectional_stack_attention_predit_model(model: Model, encoder_inputs, decoder_inputs):
    dh0_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_h0')
    dc0_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_c0')
    decoder_states0 = [dh0_input, dc0_input]
    dh1_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_h1')
    dc1_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_c1')
    decoder_states1 = [dh1_input, dc1_input]
    decoder_states_inputs = [dh0_input, dc0_input, dh1_input, dc1_input]
    encoder_each_h_input = Input(shape=(None, LSTM_DECODER_DIM*2), name='input_each_h')

    encoder_outputs, efh0, efc0, ebh0, ebc0 = model.get_layer('encoder_lstm_0').output
    state_eh0 = Concatenate()([efh0, ebh0])
    state_ec0 = Concatenate()([efc0, ebc0])
    encoder_lstm1 = model.get_layer('encoder_lstm')
    e_outputs, efh1, efc1, ebh1, ebc1 = encoder_lstm1(encoder_outputs)
    state_eh1 = Concatenate()([efh1, ebh1])
    state_ec1 = Concatenate()([efc1, ebc1])
    decoder_lstm0 = model.get_layer('decoder_lstm_0')
    decoder_outputs, dh0, dc0 = decoder_lstm0(decoder_inputs, initial_state=decoder_states0)
    decoder_lstm1 = model.get_layer('decoder_lstm')
    decoder_outputs, dh1, dc1 = decoder_lstm1(decoder_outputs, initial_state=decoder_states1)
    context = attention_section(encoder_each_h_input, decoder_outputs)
    decoder_combined_context = concatenate([context, decoder_outputs])
    dense1 = model.get_layer('dense1')
    output = dense1(decoder_combined_context)
    encoder_states = [e_outputs, state_eh0, state_ec0, state_eh1, state_ec1]
    decoder_states = [dh0, dc0, dh1, dc1]
    return encoder_states, decoder_states, output, [encoder_each_h_input]+decoder_states_inputs

def encoder_bidirectional_stack_predit_model(model: Model, encoder_inputs, decoder_inputs):
    dh0_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_h0')
    dc0_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_c0')
    decoder_states0 = [dh0_input, dc0_input]
    dh1_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_h1')
    dc1_input = Input(shape=(LSTM_DECODER_DIM*2,), name='input_c1')
    decoder_states1 = [dh1_input, dc1_input]
    decoder_states_inputs = [dh0_input, dc0_input, dh1_input, dc1_input]
    
    encoder_outputs, efh0, efc0, ebh0, ebc0 = model.get_layer('encoder_lstm_0').output
    state_eh0 = Concatenate()([efh0, ebh0])
    state_ec0 = Concatenate()([efc0, ebc0])
    encoder_lstm1 = model.get_layer('encoder_lstm')
    _, efh1, efc1, ebh1, ebc1 = encoder_lstm1(encoder_outputs)
    state_eh1 = Concatenate()([efh1, ebh1])
    state_ec1 = Concatenate()([efc1, ebc1])
    decoder_lstm0 = model.get_layer('decoder_lstm_0')
    decoder_outputs, dh0, dc0 = decoder_lstm0(decoder_inputs, initial_state=decoder_states0)
    decoder_lstm1 = model.get_layer('decoder_lstm')
    decoder_outputs, dh1, dc1 = decoder_lstm1(decoder_outputs, initial_state=decoder_states1)

    encoder_states = [state_eh0, state_ec0, state_eh1, state_ec1]
    decoder_states = [dh0, dc0, dh1, dc1]
    return encoder_states, decoder_states, decoder_outputs, decoder_states_inputs

def seq2seq_predit_model(model: Model, model_type=SeqModelType.normal.value, layer2_lstm=False)-> Model:
    encoder_inputs = model.get_layer('encoder_input').input  # input_1
    decoder_inputs = model.get_layer('decoder_input').input   # input_2
    if model_type==SeqModelType.encoder_bidirectional.value:
        if layer2_lstm:
            encoder_states, decoder_states, decoder_outputs, decoder_states_inputs = encoder_bidirectional_stack_predit_model(model, encoder_inputs, decoder_inputs)
        else:
            encoder_states, decoder_states, decoder_outputs, decoder_states_inputs = encoder_bidirectional_predit_model(model, encoder_inputs, decoder_inputs)
    elif model_type==SeqModelType.bidirectional.value:
        if layer2_lstm:
            encoder_states, decoder_states, decoder_outputs, decoder_states_inputs = bidirectional_stack_predit_model(model, encoder_inputs, decoder_inputs)
        else:
            encoder_states, decoder_states, decoder_outputs, decoder_states_inputs = bidirectional_predit_model(model, encoder_inputs, decoder_inputs)
    elif model_type==SeqModelType.normal.value:
        if layer2_lstm:
            encoder_states, decoder_states, decoder_outputs, decoder_states_inputs = normal_stack_predit_model(model, encoder_inputs, decoder_inputs)
        else:
            encoder_states, decoder_states, decoder_outputs, decoder_states_inputs = normal_predit_model(model, encoder_inputs, decoder_inputs)

    elif model_type ==SeqModelType.encoder_bidirectional_attention.value:
        if layer2_lstm:
            encoder_states, decoder_states, decoder_outputs, decoder_states_inputs = encoder_bidirectional_stack_attention_predit_model(model, encoder_inputs, decoder_inputs)
        else:
            encoder_states, decoder_states, decoder_outputs, decoder_states_inputs = encoder_bidirectional_attention_predit_model(model, encoder_inputs, decoder_inputs)
    elif model_type ==SeqModelType.normal_attention.value:
        if layer2_lstm:
            pass
        else:
            encoder_states, decoder_states, decoder_outputs, decoder_states_inputs = normal_attention_predit_model(model, encoder_inputs, decoder_inputs)
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.summary()

    print('decoder_outputs', decoder_outputs)
    decoder_dense = model.get_layer('decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    decoder_model.summary()
    return encoder_model, decoder_model


def normal_training_model(encoder_inputs, decoder_inputs):

    encoder = LSTM(LSTM_ENCODER_DIM, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    decoder_lstm = LSTM(LSTM_DECODER_DIM, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    return decoder_outputs

def normal_attention_training_model(encoder_inputs, decoder_inputs):

    encoder = LSTM(LSTM_ENCODER_DIM, return_sequences=True, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    decoder_lstm = LSTM(LSTM_DECODER_DIM, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    context = attention_section(encoder_outputs, decoder_outputs)
    decoder_combined_context = concatenate([context, decoder_outputs])
    print('decoder_combined_context', decoder_combined_context)
    output = TimeDistributed(Dense(LSTM_DECODER_DIM, activation="tanh"), name="dense1")(decoder_combined_context)
    print('output', output)
    return output

def normal_stack_training_model(encoder_inputs, decoder_inputs):
    encoder_outputs, eh0, ec0 = LSTM(LSTM_ENCODER_DIM, return_sequences=True, return_state=True, name="encoder_lstm_0")(encoder_inputs)
    _, eh1, ec1 = LSTM(LSTM_ENCODER_DIM, return_state=True, name="encoder_lstm")(encoder_outputs)
    encoder_states = [eh0, ec0, eh1, ec1]
    decoder_lstm0 = LSTM(LSTM_DECODER_DIM, return_sequences=True, return_state=True, name="decoder_lstm_0")
    decoder_outputs, dh0, dh0 = decoder_lstm0(decoder_inputs, initial_state=[eh0, ec0])
    decoder_lstm1 = LSTM(LSTM_DECODER_DIM, return_sequences=True,return_state=True, name="decoder_lstm")
    decoder_outputs, dh1, dh1 = decoder_lstm1(decoder_outputs, initial_state=[eh1, ec1])

    return decoder_outputs
    
def bidirectional_training_model(encoder_inputs, decoder_inputs):
    encoder = Bidirectional(LSTM(LSTM_ENCODER_DIM, return_state=True), name="encoder_lstm")
    encoder_outputs, efh, efc, ebh, ebc = encoder(encoder_inputs)
    encoder_states = [efh, efc, ebh, ebc]
    decoder_lstm = Bidirectional(LSTM(
        LSTM_DECODER_DIM, return_sequences=True, return_state=True), name="decoder_lstm")
    decoder_outputs, _, _, _, _, = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    return decoder_outputs

def bidirectional_stack_training_model(encoder_inputs, decoder_inputs):
    encoder_lstm0 = Bidirectional(LSTM(LSTM_ENCODER_DIM, return_sequences=True, return_state=True), name="encoder_lstm_0")
    encoder_outputs, efh0, efc0, ebh0, ebc0 = encoder_lstm0(encoder_inputs)
    encoder_lstm1 = Bidirectional(LSTM(LSTM_ENCODER_DIM, return_state=True), name="encoder_lstm")
    _, efh1, efc1, ebh1, ebc1 = encoder_lstm1(encoder_outputs)
    decoder_lstm0 = Bidirectional(LSTM(LSTM_DECODER_DIM, return_sequences=True, return_state=True), name="decoder_lstm_0")
    decoder_outputs, _, _, _, _, = decoder_lstm0(decoder_inputs, initial_state=[efh0, efc0, ebh0, ebc0])
    decoder_lstm2 = Bidirectional(LSTM(LSTM_DECODER_DIM, return_sequences=True, return_state=True), name="decoder_lstm")
    decoder_outputs, _, _, _, _, = decoder_lstm2(decoder_outputs, initial_state=[efh1, efc1, ebh1, ebc1])

    return decoder_outputs

def encoder_bidirectional_attention_training_model(encoder_inputs, decoder_inputs):
    # decoder_input = Embedding(output_dict_size, DECODER_LSTM_VEC, input_length=max_decoder_output_length, mask_zero=True, name="decoder_embedding")(decoder_input)
    encoder = Bidirectional(LSTM(LSTM_ENCODER_DIM, return_sequences=True, return_state=True), name="encoder_lstm")
    encoder_outputs, efh, efc, ebh, ebc = encoder(encoder_inputs)
    print('decoder_outputs', encoder_outputs)

    state_h = Concatenate()([efh,  ebh])
    state_c = Concatenate()([efc, ebc])
    encoder_states = [state_h, state_c]
    decoder_lstm = LSTM(LSTM_DECODER_DIM*2, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _, = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    print('decoder_outputs', decoder_outputs)
    context = attention_section(encoder_outputs, decoder_outputs)
    decoder_combined_context = concatenate([context, decoder_outputs])
    print('decoder_combined_context', decoder_combined_context)
    output = TimeDistributed(Dense(LSTM_DECODER_DIM, activation="tanh"), name='dense1')(decoder_combined_context)
    print('output', output)
    return output

def encoder_bidirectional_training_model(encoder_inputs, decoder_inputs):
    encoder = Bidirectional(LSTM(LSTM_ENCODER_DIM, return_sequences=True, return_state=True), name="encoder_lstm")
    encoder_outputs, efh, efc, ebh, ebc = encoder(encoder_inputs)
    state_h = Concatenate()([efh,  ebh])
    state_c = Concatenate()([efc, ebc])
    encoder_states = [state_h, state_c]
    decoder_lstm = LSTM(LSTM_DECODER_DIM*2, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _, = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    return decoder_outputs

def encoder_bidirectional_stack_attention_training_model(encoder_inputs, decoder_inputs):
    encoder_lstm0 = Bidirectional(LSTM(LSTM_ENCODER_DIM, return_sequences=True, return_state=True), name="encoder_lstm_0")
    encoder_outputs, efh0, efc0, ebh0, ebc0 = encoder_lstm0(encoder_inputs)
    encoder_lstm1 = Bidirectional(LSTM(LSTM_ENCODER_DIM, return_sequences=True, return_state=True), name="encoder_lstm")
    encoder_outputs, efh1, efc1, ebh1, ebc1 = encoder_lstm1(encoder_outputs)
    layer0_state_h = Concatenate()([efh0,  ebh0])
    layer0_state_c = Concatenate()([efc0,  ebc0])
    layer1_state_h = Concatenate()([efh1,  ebh1])
    layer1_state_c = Concatenate()([efc1,  ebc1])
    decoder_lstm0 = LSTM(LSTM_DECODER_DIM*2, return_sequences=True, return_state=True, name="decoder_lstm_0")
    decoder_outputs, _, _ = decoder_lstm0(decoder_inputs, initial_state=[layer0_state_h, layer0_state_c])
    decoder_lstm2 = LSTM(LSTM_DECODER_DIM*2, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm2(decoder_outputs, initial_state=[layer1_state_h, layer1_state_c])
    context = attention_section(encoder_outputs, decoder_outputs)
    decoder_combined_context = concatenate([context, decoder_outputs])
    output = TimeDistributed(Dense(LSTM_DECODER_DIM, activation="tanh"), name='dense1')(decoder_combined_context)

    return output

def encoder_bidirectional_stack_training_model(encoder_inputs, decoder_inputs):
    encoder_lstm0 = Bidirectional(LSTM(LSTM_ENCODER_DIM, return_sequences=True, return_state=True), name="encoder_lstm_0")
    encoder_outputs, efh0, efc0, ebh0, ebc0 = encoder_lstm0(encoder_inputs)
    encoder_lstm1 = Bidirectional(LSTM(LSTM_ENCODER_DIM, return_state=True), name="encoder_lstm")
    _, efh1, efc1, ebh1, ebc1 = encoder_lstm1(encoder_outputs)
    layer0_state_h = Concatenate()([efh0,  ebh0])
    layer0_state_c = Concatenate()([efc0,  ebc0])
    layer1_state_h = Concatenate()([efh1,  ebh1])
    layer1_state_c = Concatenate()([efc1,  ebc1])
    decoder_lstm0 = LSTM(LSTM_DECODER_DIM*2, return_sequences=True, return_state=True, name="decoder_lstm_0")
    decoder_outputs, _, _ = decoder_lstm0(decoder_inputs, initial_state=[layer0_state_h, layer0_state_c])
    decoder_lstm2 = LSTM(LSTM_DECODER_DIM*2, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm2(decoder_outputs, initial_state=[layer1_state_h, layer1_state_c])

    return decoder_outputs


# model_type='normal' | 'encoder_bidirectional' | 'bidirectional
def seq2seq_train_model(num_input_token, num_target_token,
                        optimizer='rmsprop', loss='categorical_crossentropy',
                        weight_path=None, gaussian_noise=1, model_type=SeqModelType.normal.value, layer2_lstm=False):

    encoder_inputs = Input(shape=(None, num_input_token), name="encoder_input")
    decoder_inputs = Input(shape=(None, num_target_token), name="decoder_input")
    _encoder_input = encoder_inputs
    if gaussian_noise != None:
        encoder_noice = GaussianNoise(1, name="gaussian_noise")(encoder_inputs)
        _encoder_input = encoder_noice
    
    if model_type == SeqModelType.bidirectional.value:
        if layer2_lstm:
            decoder_outputs = bidirectional_stack_training_model(_encoder_input, decoder_inputs)
        else:
            decoder_outputs = bidirectional_training_model(_encoder_input, decoder_inputs)

    elif model_type == SeqModelType.encoder_bidirectional.value:
        if layer2_lstm:
            decoder_outputs = encoder_bidirectional_stack_training_model(_encoder_input, decoder_inputs)
        else:
            decoder_outputs = encoder_bidirectional_training_model(_encoder_input, decoder_inputs)
    elif model_type == SeqModelType.normal.value:
        if layer2_lstm:
            decoder_outputs = normal_stack_training_model(_encoder_input, decoder_inputs)
        else:
            decoder_outputs = normal_training_model(_encoder_input, decoder_inputs)
    elif model_type == SeqModelType.encoder_bidirectional_attention.value:
        if layer2_lstm:
            decoder_outputs = encoder_bidirectional_stack_attention_training_model(_encoder_input, decoder_inputs)
            pass
        else:
            decoder_outputs = encoder_bidirectional_attention_training_model(_encoder_input, decoder_inputs)
    elif model_type == SeqModelType.normal_attention.value:
        if layer2_lstm:
            # decoder_outputs = encoder_bidirectional_stack_training_model(_encoder_input, decoder_inputs)
            pass
        else:
            decoder_outputs = normal_attention_training_model(_encoder_input, decoder_inputs)

    # decoder_dense = TimeDistributed(Dense(num_target_token, activation='softmax'),  name="decoder_dense")
    decoder_dense = Dense(num_target_token, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    # print('decoder_outputs', decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['accuracy', loss])
    model.summary()
    if weight_path:
        model.load_weights(weight_path)
    return model

def seq2seq_training(train_model: Model, encoder_input_data, decoder_input_data, decoder_target_token,
                     checkpoint_folder, analysis_saved_folder, final_model_saved_path, initial_epoch=0, enable_early_stopping=False):

    decoder_target_data = np.roll(decoder_input_data, -1, axis=1)
    decoder_target_data[:, -1] = 0
    decoder_target_data[:, -1, decoder_target_token['EOS']] = 1
    createFolder(checkpoint_folder + str(SEQ2SEQ_EPOCHES))
    mc = callbacks.ModelCheckpoint(checkpoint_folder + str(SEQ2SEQ_EPOCHES) + '\\seq2seq-weights{epoch:05d}.h5',
                                   save_weights_only=True, period=MODE_SAVE_PERIOD)
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1)

    callables = [mc, early_stopping] if enable_early_stopping else [mc]
    history = train_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                              batch_size=BATCH_SIZE,
                              epochs=SEQ2SEQ_EPOCHES,
                              validation_split=0.2,
                              initial_epoch=initial_epoch,
                              callbacks=callables)
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
    loss, acc, cate = model.evaluate(
        [encoder_input_data, decoder_input_data], decoder_target_data)
    return_str = "\nLoss: %.4f, Accuracy: %.3f%%, Cate:%.3f%%" % (loss, acc*100, cate)
    print(return_str)
    return return_str

def seq2seq_predit(encoder_model: Model, decoder_model: Model, input_seq, decoder_tokens, 
max_decoder_seq_length, result_saved_path=None):
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
    len_states_value = len(states_value)
    # print('len_states_value', len_states_value, states_value[0].shape)
    while not stop_condition:
        # print('states_value.shape', states_value.shape)    

        # print('target_seq', target_seq)
        if len_states_value == 8:
            output_tokens, fh0, fc0, bh0, bc0, fh1, fc1, bh1, bc1 = decoder_model.predict(
                [target_seq] + states_value)
        elif len_states_value == 4 or len_states_value ==5:
            output_tokens, fh, fc, bh, bc = decoder_model.predict(
                [target_seq] + states_value)
        elif len_states_value == 2:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)
        elif len_states_value == 3:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)
        # print('output_tokens', output_tokens)
        

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_decoder_tokens[sampled_token_index]
        # print('output_tokens', output_tokens)
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
        if len_states_value == 8:
            states_value = [fh0, fc0, bh0, bc0, fh1, fc1, bh1, bc1]
        elif len_states_value == 4:
            states_value = [fh, fc, bh, bc]
        elif len_states_value == 5:
            states_value = [states_value[0], fh, fc, bh, bc]
        elif len_states_value == 2:
            states_value = [h, c]
        elif len_states_value == 3:
            states_value = [states_value[0],h, c]
    if result_saved_path:
        write_file(decoded_sentence, result_saved_path, dataDim=1)
    return decoded_sentence

