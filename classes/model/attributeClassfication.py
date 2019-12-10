from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K, callbacks
import general.path as path
import general.dataType as TYPE
from general.util import createFolder, read_file, write_file, showLoss, showAccuracy


LSTM_ENCODER_DIM = 256
LSTM_DECODER_DIM = 256

MODE_SAVE_PERIOD = 50

def cnn_vgg() -> Model:
    model = Sequential()
    model.add(Conv2D(
        64, (3, 3), activation='relu', padding='same', input_shape=(112, 112, 3)))
    model.add(
        Conv2D(64, (3, 3), activation='relu'))  # 112*112*64
    model.add(MaxPooling2D((2, 2)))
    print('MP1: ', model.output_shape)
    model.add(
        Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(
        Conv2D(128, (3, 3), activation='relu'))  # 56*56*128
    model.add(MaxPooling2D((2, 2)))
    print('MP2: ', model.output_shape)
    model.add(
        Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(
        Conv2D(256, (3, 3), activation='relu'))  # 28*28*256
    model.add(MaxPooling2D((2, 2)))  # 14*14*256
    print('MP3: ', model.output_shape)
    model.add(Flatten())
    print('flatten: ', model.output_shape)
    output_shape = model.output_shape
    model.add(Reshape((int(output_shape[1]/256), 256)))
    print(model.output_shape)
    return model


def cnn_switch(cnnType='VGG'):
    if cnnType=='VGG':
        return cnn_vgg()

def attribute_classfication_model(num_target_token, weight_path=None):
   # First, let's define a vision model using a Sequential model.
    # This model will encode an image into a vector.
    vision_model = cnn_switch('VGG')

    # Now let's get a tensor with the output of our vision model:
    image_input = Input(shape=(112, 112, 3))
    encoded_image = vision_model(image_input)

    # Next, let's define a language model to encode the question into a vector.
    # Each question will be at most 100 words long,
    # and we will index words as integers from 1 to 9999.

    encoder = LSTM(LSTM_ENCODER_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoded_image)
    encoder_states = [state_h, state_c]
    # encoder_model = Model(encoder_inputs, encoder_states)
    decoder_inputs = Input(shape=(None, num_target_token))
    decoder_lstm = LSTM(
        LSTM_DECODER_DIM, return_sequences=True, return_state=True)

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_target_token, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    attr_classfy_training_model = Model(
        [image_input, decoder_inputs], decoder_outputs)
    attr_classfy_training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                                        metrics=['accuracy'])
    
    encoder_model = Model(image_input, encoder_states)
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

    return attr_classfy_training_model, vision_model, decoder_model


def attribute_classfication_training(data_config, model: Model, epochs):
    '''
    data_config = {
        bath_size=16,
        target_size=(112, 112)
        classes: [... ]
        train: {
            folder: 'data/train/',
            total_size: 1000,
        },
        valid: { ... },
        test: { ... },
    }
    '''
    # create generator
    datagen = ImageDataGenerator(rescale=1./255)
    # prepare an iterators for each dataset
    train_it = datagen.flow_from_directory(
        data_config['train']['folder'],
        batch_size=data_config['bath_size'], class_mode=None,
        target_size=data_config['target_size']
    )
    val_it = datagen.flow_from_directory(
        data_config['valid']['folder'],
        batch_size=data_config['bath_size'], class_mode=None,
        target_size=data_config['target_size']
    )
    test_it = datagen.flow_from_directory(
        data_config['test']['folder'],
        batch_size=data_config['bath_size'], class_mode=None,
        target_size=data_config['target_size']
    )

    mc = callbacks.ModelCheckpoint(path.CLASS_ATTR_WEIGHT + str(epochs) + '\\' + 'attr-weights{epoch:05d}.h5',
                                   save_weights_only=True, period=MODE_SAVE_PERIOD)
    train_step_per_each = data_config['train']['total_size'] // data_config['bath_size']
    valid_step_per_each = data_config['train']['total_size'] // data_config['bath_size']
    history = model.fit_generator(train_it,
                                  steps_per_epoch=train_step_per_each,
                                  epochs=epochs,
                                  validation_data=val_it,
                                  validation_steps=valid_step_per_each,
                                  callbacks=[mc]
                                  )

    showLoss(history, path.CLASS_ATTR_ANALYSIS_PATH, 'loss-'+ str(epochs))
    showAccuracy(history, path.CLASS_ATTR_ANALYSIS_PATH, 'accuracy-'+ str(epochs))
    write_file(history.history, path.CLASS_SEQ2SEQ_ANALYSIS_PATH, 'history'+str(epochs), TYPE.TXT, 'JSON')
    
    
