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


def cnn_vgg(weight_path=None) -> Model:
    model = Sequential(name='vision_model')
    model.add(Conv2D(
        64, (3, 3), activation='relu', padding='same', input_shape=(112, 112, 3)))
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
    output_shape = model.output_shape
    model.add(Reshape((int(output_shape[1]/256), 256), name='cnn_output'))
    model.summary()
    if weight_path:
        model.load_weights(weight_path)
    return model


def get_cnn_model(cnnType='VGG', pre_trained_weight=None)-> Model:
    if cnnType == 'VGG':
        return cnn_vgg(pre_trained_weight)


def attribute_classification_train_model(num_target_token,  cnn_model='VGG', input_shape=(112, 112, 3),
                                         cnn_model_weight_path=None, optimizer='rmsprop', loss='categorical_crossentropy',
                                         weight_path=None):
    vision_model = get_cnn_model(cnn_model, cnn_model_weight_path)
    image_input = Input(shape=input_shape, name='image_input')
    encoded_image = vision_model(image_input)
    encoder = LSTM(LSTM_ENCODER_DIM, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder(encoded_image)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None, num_target_token), name='decoder_input')
    decoder_lstm = LSTM(
        LSTM_DECODER_DIM, return_sequences=True, return_state=True, name='decoder_lstm')

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_target_token, activation='softmax', name='decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)
    train_model = Model(
        [image_input, decoder_inputs], decoder_outputs)
    train_model.compile(optimizer=optimizer, loss=loss,
                        metrics=['accuracy'])
    train_model.summary()
    return train_model


def attribute_classification_predit_model(model: Model)-> Model:
    encoder_inputs = model.get_layer('image_input')  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.summary()
    decoder_inputs = model.get_layer('decoder_input')   # input_2
    decoder_state_input_h = Input(shape=(LSTM_DECODER_DIM,), name='decoder_state_input_h')
    decoder_state_input_c = Input(shape=(LSTM_DECODER_DIM,), name='decoder_state_input_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = model.get_layer('decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    decoder_model.summary()
    return encoder_model, decoder_model


def attribute_classfication_model(num_target_token, weight_path=None):
   # First, let's define a vision model using a Sequential model.
    # This model will encode an image into a vector.
    vision_model = get_cnn_model('VGG')

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


def attribute_classfication_training(train_model: Model, data_config,  epochs, decoder_input_data):
    '''
    data_config = {
        bath_size=16,
        target_size=(112, 112)
        decoder_target_tokens: [... ]
        image: {
            train: {
                folder: 'data/train/',
                total_size: 1000,
                decoder_input_data: 
            },
            valid: { ... },
            test: { ... },
        },
        checkpoint_folder: "", 
        analysis_saved_folder: "",
        final_model_saved_path: "",
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

    mc = callbacks.ModelCheckpoint(data_config['checkpoint_folder'] + str(epochs) + '\\' + 'attr-weights{epoch:05d}.h5',
                                   save_weights_only=True, period=MODE_SAVE_PERIOD)
    train_step_per_each = data_config['train']['total_size'] // data_config['bath_size']
    valid_step_per_each = data_config['train']['total_size'] // data_config['bath_size']
    history = train_model.fit_generator(datagen.flow(),
                                  steps_per_epoch=train_step_per_each,
                                  epochs=epochs,
                                  validation_data=val_it,
                                  validation_steps=valid_step_per_each,
                                  callbacks=[mc]
                                  )

    showLoss(history, data_config['analysis_saved_folder'], 'loss-' + str(epochs))
    showAccuracy(history, data_config['analysis_saved_folder'],
                 'accuracy-' + str(epochs))
    write_file(history.history, data_config['analysis_saved_folder']+'history'+str(epochs)+TYPE.TXT, 'JSON')
    train_model.save(data_config['final_model_saved_path'])
    return train_model