from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
from keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Dropout, RepeatVector
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K, callbacks
import general.path as path
import general.dataType as TYPE
from general.util import createFolder, read_file, write_file, showLoss, showAccuracy
from classes.data2Input import attributes_data_generator, preprocess_image, decoder_tokens_list_to_dict
import numpy as np
from PIL import Image

MAX_DECODER_INPUT_LENGTH = 4
LSTM_ENCODER_DIM = 256
LSTM_DECODER_DIM = 256

MODE_SAVE_PERIOD = 100
EPOCHES = 100
BATCH_SIZE = 8


def cnn_simple_vgg(input_shape,weight_path=None) -> Model:
    model = Sequential(name='vision_model')
    model.add(Conv2D(
        64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(
        Conv2D(64, (3, 3), activation='relu'))  # 112*112*64
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(
        Conv2D(128, (3, 3), activation='relu'))  # 56*56*128
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(
        Conv2D(256, (3, 3), activation='relu'))  # 28*28*256
    model.add(MaxPooling2D((2, 2)))  # 14*14*256
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(LSTM_ENCODER_DIM, activation='relu'))
    model.add(Dropout(0.3))
    model.add(RepeatVector(MAX_DECODER_INPUT_LENGTH))

    # output_shape = model.output_shape
    # model.add(Reshape((int(output_shape[1]/256), 256), name='cnn_output'))
    model.summary()
    if weight_path:
        model.load_weights(weight_path)
    return model

def cnn_VGG16(input_shape=(224,224,3),weight_path=None) -> Model:
     
    model = Sequential()
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(LSTM_ENCODER_DIM,activation='softmax'))
    model.add(RepeatVector(MAX_DECODER_INPUT_LENGTH))

    model.summary()

    if weight_path:
        model.load_weights(weight_path)
    return model 

def cnn_alexnet(input_shape=(227,227,3),weight_path=None) -> Model:

    seed = 7
    np.random.seed(seed)
    
    model = Sequential()  #input_shape=(227,227,3)
    model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=input_shape,padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(LSTM_ENCODER_DIM,activation='softmax')) #1000
    model.add(RepeatVector(MAX_DECODER_INPUT_LENGTH))

    # model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.summary()
    if weight_path:
        model.load_weights(weight_path)

    return model

#  CNN MODELS:https://blog.csdn.net/wmy199216/article/details/71171401
def get_cnn_model(cnnType='VGG16', input_shape=(112, 112, 3), pre_trained_weight=None)-> Model:
    if cnnType == 'simple_VGG':
        return cnn_simple_vgg(input_shape, pre_trained_weight)
    if cnnType == 'Alexnet':
        return cnn_alexnet(input_shape, pre_trained_weight)
    if cnnType == 'VGG16':
        return cnn_VGG16(input_shape, pre_trained_weight)


def attribute_classification_train_model(num_target_token,  cnn_model='VGG', input_shape=(112, 112, 3),
                                         cnn_model_weight_path=None, optimizer='rmsprop', loss='categorical_crossentropy',
                                         weight_path=None):
    vision_model = get_cnn_model(cnn_model, input_shape, cnn_model_weight_path)
    image_input = Input(shape=input_shape, name='image_input')
    encoded_image = vision_model(image_input)
    encoder = LSTM(LSTM_ENCODER_DIM, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder(encoded_image)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(
        shape=(None, num_target_token), name='decoder_input')
    decoder_lstm = LSTM(
        LSTM_DECODER_DIM, return_sequences=True, return_state=True, name='decoder_lstm')

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(
        num_target_token, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    train_model = Model(
        [image_input, decoder_inputs], decoder_outputs)
    train_model.compile(optimizer=optimizer, loss=loss,
                        metrics=['accuracy'])
    train_model.summary()
    return train_model


def attribute_classification_predit_model(model: Model)-> Model:
    encoder_inputs = model.get_layer('image_input').input  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer(
        'encoder_lstm').output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.summary()
    decoder_inputs = model.get_layer('decoder_input').input   # input_2
    decoder_state_input_h = Input(
        shape=(LSTM_DECODER_DIM,), name='decoder_state_input_h')
    decoder_state_input_c = Input(
        shape=(LSTM_DECODER_DIM,), name='decoder_state_input_c')
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


def attribute_classfication_training(train_model: Model,  encoder_config, decoder_config,
                                     checkpoint_folder, analysis_saved_folder, final_model_saved_path, initial_epoch=0, keep_ratio=True):
    mc = callbacks.ModelCheckpoint(checkpoint_folder + str(EPOCHES) + 'attr-classfy-weights{epoch:05d}.h5',
                                   save_weights_only=True, period=MODE_SAVE_PERIOD)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    lines = read_file(decoder_config['data_path'], 'splitlines')
    num_train = encoder_config['num_train']
    num_valid = encoder_config['num_valid']
    token_list = decoder_config['token_list']
    input_shape = encoder_config['input_shape']
    # print('-----config----- \nnum_train: {}\nnum_valid: {}\ninput_shape: {}\nsteps_per_epoch: {}\n'.format(num_train, num_valid, input_shape, max(1, num_train//BATCH_SIZE)))
    history = train_model.fit_generator(attributes_data_generator(lines[:num_train],BATCH_SIZE, input_shape, token_list, keep_ratio=keep_ratio),
                                        steps_per_epoch=max(1, num_train//BATCH_SIZE),
                                        validation_data=attributes_data_generator(lines[num_train:num_train + num_valid],BATCH_SIZE, input_shape, token_list),
                                        validation_steps=max(1, num_valid//BATCH_SIZE),
                                        epochs=EPOCHES,
                                        initial_epoch=initial_epoch,
                                        callbacks=[mc, early_stopping])

    showLoss(history, analysis_saved_folder, 'loss' + str(EPOCHES))
    showAccuracy(history, analysis_saved_folder,
                 'accuracy' + str(EPOCHES))
    write_file(history.history, analysis_saved_folder +
               'history'+str(EPOCHES)+TYPE.TXT, 'JSON')
    train_model.save(final_model_saved_path)
    return train_model



def attribute_classification_predit(encoder_model: Model, decoder_model: Model, 
input_image_path, input_shape, decoder_token_list, max_decoder_seq_length, result_saved_path=None, img_input_type='path'):
    img_data = preprocess_image(input_image_path, input_shape, img_input_type=img_input_type)
    w, h, c = img_data.shape
    img_input = np.zeros((1, w, h, c), dtype='float32')
    img_input[0] = img_data
    states_value = encoder_model.predict(img_input)
    tokens_dict = decoder_tokens_list_to_dict(decoder_token_list)
    target_seq = np.zeros((1, 1, len(tokens_dict)))
    target_seq[0, 0, tokens_dict['START']] = 1.

    reverse_tokens_dict = dict(
        (i, token) for token, i in tokens_dict.items())
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_tokens_dict[sampled_token_index]
        if sampled_token != 'EOS':
            decoded_sentence.append(sampled_token)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_token == 'EOS' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, len(tokens_dict)))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]
    if result_saved_path:
        write_file(decoded_sentence, result_saved_path, dataDim=1)
    return decoded_sentence


def attribute_classification_evaluate(model: Model, start_idx, end_idx, input_shape, decoder_config):
    lines = read_file(decoder_config['data_path'], 'splitlines')
    token_list = decoder_config['token_list']
    loss, acc = model.evaluate_generator(attributes_data_generator(lines[start_idx:end_idx],BATCH_SIZE, input_shape, token_list),
    steps=max(1, (end_idx - start_idx)//BATCH_SIZE))
    res = "\nLoss: %.4f, Accuracy: %.3f%% \n " % (loss, acc*100)
    print(res)
    return res
