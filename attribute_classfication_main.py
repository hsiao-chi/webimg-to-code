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
from general.node.nodeEnum import Font_color
from keras.models import load_model
import random
from evaluationCode.heatmap import compare_attr_class, show_heatmap
from general.node.nodeEnum import Font_color, Bg_color


if __name__ == "__main__":
    DEBUG_DATASET = False
    TRAINING = True
    PREDIT = True
    EVALUATE = True
    HEATMAP =True

    keep_img_ratio=True
    cnn_model = 'simple_VGG'
    dataset = 'data3'
    eva_record_path = path.EVALUATION_ATTR_CLASS_EVALUATION+dataset+"\\"
    eva_record_name = 'simple_VGG(74-224-256-e100)-noise-e100-norepeat.txt'
    predit_data_path = path.SELF+'test-predit\\attr-class-predit\\'+dataset+"\\"
    predit_data_name = 'simple_VGG(74-224-256-e100)-noise-e100-norepeat'
    predit_data_start_idx = 2800
    predit_data_num = 300
    final_model_saved_path = path.CLASS_ATTR_MODEL_PATH + str(EPOCHES)+'\\attr_class_model'+TYPE.H5
    # predit_model_path = r'E:\projects\NTUST\webimg-to-code\assets\attr_class-data3\test\simple-VGG\74-112-256\p0\model\100\attr_class_model.h5'
    predit_model_path = final_model_saved_path
    evaluate_model_path = final_model_saved_path

    encoder_config = get_attribute_encoder_config(2)
    decoder_config = get_attribute_decoder_config(2)
    lines = read_file(decoder_config['data_path'], 'splitlines')

    if TRAINING:
        # weight_path=r'E:\projects\NTUST\webimg-to-code\assets\attr_class-pix2code\test\simple-VGG\74-112-256\p0\model\22\attr_class_model.h5'
        weight_path=None
        createFolder(path.CLASS_ATTR_MODEL_PATH + str(EPOCHES))
        createFolder(path.CLASS_ATTR_WEIGHT + str(EPOCHES))
        train_model = attribute_classification_train_model(len(decoder_config['token_list']),
                                                           input_shape=encoder_config['input_shape'], optimizer='Adadelta'
                                                           , cnn_model=cnn_model, weight_path=weight_path)
        attribute_classfication_training(train_model, encoder_config, decoder_config,
                                         checkpoint_folder=path.CLASS_ATTR_WEIGHT + str(EPOCHES), analysis_saved_folder=path.CLASS_ATTR_ANALYSIS,
                                         final_model_saved_path=final_model_saved_path, initial_epoch=0,
                                         keep_ratio=keep_img_ratio)

    if EVALUATE:
        str_model_path = 'evaluated Model path: \n{}'.format(evaluate_model_path)
        str_data_file = '\ndata_file: {}'.format(decoder_config['data_path'])
        print(str_model_path)
        print(str_data_file)
        test_start = encoder_config['num_train']+encoder_config['num_valid']
        test_end = test_start+encoder_config['num_test']
        input_shape = encoder_config['input_shape']
        str_training_data='\n\ntraining data: \n from: {}\n to: {}'.format(0, test_start)
        print(str_training_data)
        str_train_result = attribute_classification_evaluate(load_model(evaluate_model_path), 
        0, test_start, input_shape, decoder_config)
        str_testing_data='\n\ntesting data: \n from: {}\n to: {}'.format(test_start, test_end)
        print(str_testing_data)
        str_test_result = attribute_classification_evaluate(load_model(evaluate_model_path), 
        test_start, test_end, input_shape, decoder_config)
        res = str_model_path+str_data_file+str_training_data+str_train_result+str_testing_data+str_test_result
        createFolder(eva_record_path)
        write_file(res, eva_record_path+eva_record_name, dataDim=0)
    
    if PREDIT:
        createFolder(path.CLASS_ATTR_PREDIT_GUI_PATH + str(EPOCHES))
        encoder_model, decoder_model = attribute_classification_predit_model(
            load_model(predit_model_path))
        max_data_length = len(lines)
        predit_list = []
        for i in range(predit_data_start_idx, predit_data_start_idx+predit_data_num):
            line = lines[i].split()
            # print('origin:', line)
            decoded_sentence = attribute_classification_predit(encoder_model, decoder_model, line[0], encoder_config['input_shape'], decoder_config['token_list'], 4)                                                   
            # print('decoded_sentence: ', i, decoded_sentence) 
            predit_list.append([line[0]]+decoded_sentence)
        if HEATMAP:
            createFolder(predit_data_path)
            predit_file_name = predit_data_path+predit_data_name+TYPE.TXT
            write_file(predit_list, predit_file_name, dataDim=2)
            labels = decoder_config['token_list']
            labels.remove('START')
            # labels.remove(Font_color.success.value)
            # labels.remove(Font_color.danger.value)
            # labels.remove('EOS')
            target = compare_attr_class(decoder_config['data_path'],predit_file_name,  labels, labels)
            show_heatmap(target, labels, labels, ratio=True, 
            save_path=eva_record_path+predit_data_name)

    
    if DEBUG_DATASET:
        for line in lines:
            line = line.split()
            try: 
                preprocess_image(line[0], (112,112,3), True)
            except ValueError:
                print('ERROR File: {}'.format(line))
                pass
