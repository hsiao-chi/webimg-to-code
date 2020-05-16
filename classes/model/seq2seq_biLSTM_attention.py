from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, GaussianNoise, Bidirectional, concatenate, Concatenate, Activation, dot, TimeDistributed, Embedding
from keras import backend as K, callbacks
import numpy as np
from classes.data2Input import to_Seq2Seq_encoder_input, encoder_tokens_list_to_dict, decoder_tokens_list_to_dict
from general.util import createFolder, write_file, showLoss, showAccuracy
import general.dataType as TYPE

ENCODER_LSTM_VEC = 64
DECODER_LSTM_VEC = 64
WORD_EMBD_VEC = 64
SEQ2SEQ_EPOCHES = 10
MODE_SAVE_PERIOD = 100
BATCH_SIZE=64

def lstm_attention_model(num_encoder_input_vec: int, max_decoder_output_length=300, output_dict_size=19)->Model:
    encoder_input = Input(shape=(None, num_encoder_input_vec), name="encoder_input")
    decoder_input = Input(shape=(max_decoder_output_length,), name="decoder_input")
    encoder = GaussianNoise(1, name="gaussian_noise")(encoder_input)
    encoder = LSTM(ENCODER_LSTM_VEC, return_sequences=True, name="encoder_lstm")(encoder_input)
    encoder_last = encoder[:,-1,:]
    print('encoder', encoder)
    print('encoder_last', encoder_last)
    decoder = Embedding(output_dict_size, DECODER_LSTM_VEC, input_length=max_decoder_output_length, mask_zero=True, name="decoder_embedding")(decoder_input)
    decoder = LSTM(DECODER_LSTM_VEC, return_sequences=True, name="decoder_lstm")(decoder, initial_state=[encoder_last, encoder_last])

    attention = dot([decoder, encoder], axes=[2, 2], name="dot1")
    attention = Activation('softmax', name='attention')(attention)
    print('attention', attention)

    context = dot([attention, encoder], axes=[2,1], name="dot2")
    print('context', context)

    decoder_combined_context = Concatenate(name="decoder_combined_context")([context, decoder])
    print('decoder_combined_context', decoder_combined_context)

    # Has another weight + tanh layer as described in equation (5) of the paper
    output = TimeDistributed(Dense(64, activation="tanh"), name="dense1")(decoder_combined_context)
    output = TimeDistributed(Dense(output_dict_size, activation="softmax"), name="dense2")(output)
    print('output', output)
    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def mapping_gui_skeleton_token_index(seqs, token_list: list, max_len):
    token_dict = decoder_tokens_list_to_dict(token_list, 1)
    print('token_dict', token_dict)
    temp=[]
    _max_len = max_len
    # eos = token_list.index('EOS')
    for seq in seqs:
        t = [token_dict[t] for t in seq]
        t = [token_dict['START']]+t+[token_dict['EOS']]
        _max_len = max(_max_len, len(t))
        temp.append(t)
    target = np.zeros((len(temp), _max_len))
    for idx in range(len(temp)):
        for j in range(_max_len):
            try:
                target[idx, j] = temp[idx][j]
            except IndexError:
                target[idx, j] = token_dict['EOS']
    return target

def to_batch_encoder_input(input_seqs_list: list, encoder_config, max_len=50) -> np.array:
    encoder_tokens = encoder_tokens_list_to_dict(encoder_config['token_list'], encoder_config['class_mode'])
    _max_len=max_len
    full_data=[]
    for input_seqs in input_seqs_list:
        temp_data = []
        _max_len = max(_max_len, len(input_seqs))
        for input_seq in input_seqs:
            data = input_seq[:encoder_config['direct_part']]
            attrs = [0]*len(encoder_tokens)
            for attr in input_seq[encoder_config['direct_part']:]:
                if encoder_config['class_mode']:
                    for idx, target_list in enumerate(encoder_tokens):
                        try:
                            attrs[idx] = target_list[attr]
                        except KeyError:
                            pass
                else:
                    attrs[encoder_tokens[attr]] = 1
            temp_data.append(data+attrs)
        full_data.append(temp_data)

    encoder_input_data = np.zeros((len(input_seqs_list), _max_len, len(encoder_tokens)+encoder_config['direct_part']), dtype='float32')
    for d, data in enumerate(full_data):
        for i, line in enumerate(data):
            encoder_input_data[d, i] = line
    return encoder_input_data


def lstm_attention_training(model: Model, encoder_input_list: list, decoder_input_list: list, 
encoder_config, decoder_config, checkpoint_folder, analysis_saved_folder, final_model_saved_path, encoder_max_len=50, decoder_max_len=250):
    '''
    @ param encoder_input_list: list, 3dim,(batch_size, input_length, 5~7), attribute還是attribute 座標還是座標

    @ param decoder_input_list: 2dim, (batch_size, gui_skeleton_list)不含'START', 'EOS'
    
    @ param attributes_dict: [attributes]

    '''
    training_encoder_input=to_batch_encoder_input(encoder_input_list, encoder_config, encoder_max_len)
    # print('training_encoder_input: ', training_encoder_input.shape, '\n', training_encoder_input)

    training_decoder_input = mapping_gui_skeleton_token_index(decoder_input_list, decoder_config['token_list'], decoder_max_len)
    _training_decoder_output = np.roll(training_decoder_input, -1, axis=1)
    _training_decoder_output[:, -1] = _training_decoder_output[:, -2]
    training_decoder_output=np.eye(len(decoder_config['token_list'])+1)[_training_decoder_output.astype('int')]

    print('training_decoder_input: ', training_decoder_input.shape, '\n', training_decoder_input)
    print('_training_decoder_output: ', _training_decoder_output.shape, '\n', _training_decoder_output)
    print('training_decoder_output: ', training_decoder_output.shape, '\n', training_decoder_output)

    mc = callbacks.ModelCheckpoint(checkpoint_folder + str(SEQ2SEQ_EPOCHES) + '\\seq2seq-weights{epoch:05d}.h5',
                                   save_weights_only=True, period=MODE_SAVE_PERIOD)
    history = model.fit([training_encoder_input, training_decoder_input], training_decoder_output,
                              batch_size=BATCH_SIZE,
                              epochs=SEQ2SEQ_EPOCHES,
                              validation_split=0.2,
                              initial_epoch=0,
                              callbacks=[mc])
    showLoss(history, analysis_saved_folder, 'loss' + str(SEQ2SEQ_EPOCHES))
    showAccuracy(history, analysis_saved_folder,
                 'accuracy' + str(SEQ2SEQ_EPOCHES))
    write_file(history.history, analysis_saved_folder +
               'history'+str(SEQ2SEQ_EPOCHES)+TYPE.TXT, 'JSON')
    model.save(final_model_saved_path)
    return model


def generate(model: Model, encoder_input_list, encoder_config, max_output_len, gui_token_dict: list, encoder_max_len=50, decoder_max_len=250):
    decoder_token_dict = decoder_tokens_list_to_dict(gui_token_dict, 1)
    encoder_input = to_batch_encoder_input(encoder_input_list, encoder_config, max_len=encoder_max_len)
    decoder_input = np.zeros(shape=(len(encoder_input), max_output_len))
    decoder_input[:,0] = decoder_token_dict['START']
    # eos = gui_token_dict.index('EOS')
    # decoder_input[:,1:] = eos
    for i in range(1,max_output_len):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
        if output[:,i] == decoder_token_dict['EOS']: 
            break
    decoder_output=[]
    print('decoder_input', decoder_input)
    for i in range(max_output_len):
        if decoder_input[0][i] != decoder_token_dict['EOS']:
            decoder_output.append(gui_token_dict[decoder_input[0][i].astype('int')])
        else:
            break
    return decoder_output

# def evaluation()
