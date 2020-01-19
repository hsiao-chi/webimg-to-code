from classes.model.attributeClassfication import (
    attribute_classification_train_model,
    attribute_classfication_training,
    attribute_classification_predit_model,
    attribute_classification_predit,
    attribute_classification_evaluate,
    EPOCHES
)
from classes.data2Input import attributes_data_generator, preprocess_image
from general.util import read_file, write_file, createFolder
from classes.get_configs import get_attribute_encoder_config, get_attribute_decoder_config
import general.path as path
import general.dataType as TYPE
from keras.models import load_model
import random

if __name__ == "__main__":
    DEBUG_DATASET = False
    TRAINING = True
    PREDIT = True
    EVALUATE = True
    final_model_saved_path = path.CLASS_ATTR_MODEL_PATH + \
        str(EPOCHES)+'\\attr_class_model'+TYPE.H5
    predit_model_path = final_model_saved_path
    evaluate_model_path = final_model_saved_path

    encoder_config = get_attribute_encoder_config(1)
    decoder_config = get_attribute_decoder_config(1)
    lines = read_file(decoder_config['data_path'], 'splitlines')

    if TRAINING:

        createFolder(path.CLASS_ATTR_MODEL_PATH + str(EPOCHES))
        createFolder(path.CLASS_ATTR_WEIGHT + str(EPOCHES))
        train_model = attribute_classification_train_model(len(decoder_config['token_list']),
                                                           input_shape=encoder_config['input_shape'])
        attribute_classfication_training(train_model, encoder_config, decoder_config,
                                         checkpoint_folder=path.CLASS_ATTR_WEIGHT + str(EPOCHES), analysis_saved_folder=path.CLASS_ATTR_ANALYSIS,
                                         final_model_saved_path=final_model_saved_path, initial_epoch=0)

    if PREDIT:
        createFolder(path.CLASS_ATTR_PREDIT_GUI_PATH + str(EPOCHES))
        encoder_model, decoder_model = attribute_classification_predit_model(
            load_model(predit_model_path))
        max_data_length = len(lines)
        for i in range(5):
            idx = random.randint(0, max_data_length+1)
            print('predit_GT: ', lines[idx])
            line = lines[idx].split()
            try:
                decoded_sentence = attribute_classification_predit(encoder_model, decoder_model, line[0], encoder_config['input_shape'], decoder_config['token_list'], 4,
                                                               result_saved_path=path.CLASS_ATTR_PREDIT_GUI_PATH + str(EPOCHES)+'\\'+str(idx)+TYPE.GUI)
                print('decoded_sentence length: ', idx, len(decoded_sentence))
            except ValueError:
                print('Predit ERROR File: {}'.format(line[0]))
                pass

    if EVALUATE:
        print('evaluated Model path: \n{}'.format(evaluate_model_path))
        print('data_file: {}'.format(decoder_config['data_path']))
        test_start = encoder_config['num_train']+encoder_config['num_valid']
        test_end = test_start+encoder_config['num_test']
        input_shape = encoder_config['input_shape']

        print('training data: \n from: {}\n to: {}'.format(0, test_start))
        attribute_classification_evaluate(load_model(evaluate_model_path), 
        0, test_start, input_shape, decoder_config)
        print('testing data: \n from: {}\n to: {}'.format(test_start, test_end))
        attribute_classification_evaluate(load_model(evaluate_model_path), 
        test_start, test_end, input_shape, decoder_config)

    
    if DEBUG_DATASET:
        for line in lines:
            line = line.split()
            try: 
                preprocess_image(line[0], (112,112,3), True)
            except ValueError:
                print('ERROR File: {}'.format(line))
                pass
