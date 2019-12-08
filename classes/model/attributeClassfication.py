from keras.layers import Conv2D, MaxPooling2D, Flatten,Reshape
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
LSTM_ENCODER_DIM = 256
LSTM_DECODER_DIM = 256
def attribute_classfication_model(num_target_token, weight_path=None):
   # First, let's define a vision model using a Sequential model.
    # This model will encode an image into a vector.
    vision_endoder_model = Sequential()
    vision_endoder_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(112, 112, 3)))
    vision_endoder_model.add(Conv2D(64, (3, 3), activation='relu')) # 112*112*64
    vision_endoder_model.add(MaxPooling2D((2, 2)))
    print('MP1: ',vision_endoder_model.output_shape)
    vision_endoder_model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
    vision_endoder_model.add(Conv2D(128, (3, 3), activation='relu'))  #56*56*128
    vision_endoder_model.add(MaxPooling2D((2, 2)))
    print('MP2: ',vision_endoder_model.output_shape)
    vision_endoder_model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
    vision_endoder_model.add(Conv2D(256, (3, 3), activation='relu'))
    vision_endoder_model.add(Conv2D(256, (3, 3), activation='relu')) # 28*28*256
    vision_endoder_model.add(MaxPooling2D((2, 2)))  # 14*14*256
    print('MP3: ',vision_endoder_model.output_shape)
    vision_endoder_model.add(Flatten())
    print('flatten: ',vision_endoder_model.output_shape)
    output_shape = vision_endoder_model.output_shape
    vision_endoder_model.add(Reshape((int(output_shape[1]/256), 256)))
    print(vision_endoder_model.output_shape)

    # Now let's get a tensor with the output of our vision model:
    image_input = Input(shape=(112, 112, 3))
    encoded_image = vision_endoder_model(image_input)

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
    attr_classfy_training_model = Model([image_input, decoder_inputs], decoder_outputs)
    attr_classfy_training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

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

    return attr_classfy_training_model, vision_endoder_model, decoder_model


def attribute_classfication_training(parameter_list):
    pass