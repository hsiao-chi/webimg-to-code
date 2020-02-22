from classes.get_configs import get_encoder_config, get_decoder_config, get_attribute_encoder_config, get_attribute_decoder_config
from classes.data2Input import to_Seq2Seq_input, decoder_tokens_list_to_dict, to_Seq2Seq_encoder_input
import numpy as np
from classes.model.yolo.yolo import YOLO, detect_video
from PIL import Image
from keras.models import load_model
from classes.model.layoutGenerator import seq2seq_predit_model, seq2seq_predit, SEQ2SEQ_EPOCHES
from classes.model.attributeClassfication import attribute_classification_predit_model, attribute_classification_predit
import general.path as path
import general.dataType as TYPE
from datasetCode.dataset_2_generator.nodeTreeGenerateClass import NodeTreeGenerator
from datasetCode.dataset_2_generator.compiler import Compiler
from datasetCode.data_transform.web_to_screenshot import webscreenshoot
import cv2
from datasetCode.data_transform.tag_for_yolo import splitImage


def detect_img(yolo, img_name) -> list:
    try:
        image = Image.open(img_name)
    except:
        print('Open Error! Try again!')
    else:
        r_image, r_targets = yolo.detect_image(image)
        # r_image.show()
    # yolo.close_session()
    return r_targets


if __name__ == "__main__":
    LAYOUT_INPUT_TYPE = 2
    LAYOUT_TARGET_TYPE = 3
    USE_EPOCH = 300

    yolo_model_name = 'trained_weights_final(500-simple).h5'
    yolo_classes_name = 'pix2code_simple_classes.txt'
    test_image_path = r"test-predit\1706\1706.png"
    test_target_folder = 'E:\\projects\\NTUST\\webimg-to-code\\test-predit\\1706\\arch2\\'
    test_target_path = test_target_folder + '500-BN'
    layout_model_path = r'E:\projects\NTUST\webimg-to-code\assets\2020\seq2seq-pix2code\attr-rowcolAttrElement\2500\bidirectional-resort\model\300\model.h5'
    bidirectional_lstm = True
    attr_class_model_path = r'E:\projects\NTUST\webimg-to-code\assets\2020\attr_class-pix2code\remote\attr_class_model(simple_112_256_2300).h5'

    layout_encoder_config = get_encoder_config(LAYOUT_INPUT_TYPE)
    layout_decoder_config = get_decoder_config(LAYOUT_TARGET_TYPE)
    attr_encoder_config = get_attribute_encoder_config(1)
    attr_decoder_config = get_attribute_decoder_config(1)

    targets = detect_img(YOLO(model_name=yolo_model_name,
                              classes_name=yolo_classes_name), test_image_path)
    print(targets, '\n\n')

    full_image = cv2.imread(test_image_path)
    attr_encoder_model, attr_decoder_model = attribute_classification_predit_model(
        load_model(attr_class_model_path))
    attr_targets = []
    max_dim3_len = 0
    for idx, target in enumerate(targets):
        sub_img = splitImage(full_image, target)
        sub_img = Image.fromarray(cv2.cvtColor(sub_img,cv2.COLOR_BGR2RGB)) 
        decoded_sentence = attribute_classification_predit(
            attr_encoder_model, attr_decoder_model, sub_img, attr_encoder_config['input_shape'], attr_decoder_config['token_list'], 4, img_input_type='img')
        attr_targets.append(target+decoded_sentence)
        if max_dim3_len < len(target+decoded_sentence):
            max_dim3_len = len(target+decoded_sentence)
        print(decoded_sentence)
    print('attr_targets\n', attr_targets)
    
    layout_input_seq = to_Seq2Seq_encoder_input(attr_targets, layout_encoder_config)
    decoder_target_tokens = decoder_tokens_list_to_dict(layout_decoder_config['token_list'])
    max_decoder_len = 200

    encoder_model, decoder_model = seq2seq_predit_model(load_model(layout_model_path), bidirectional_lstm=bidirectional_lstm)
    decoded_sentence = seq2seq_predit(encoder_model, decoder_model, 
    input_seq=layout_input_seq, 
    decoder_tokens=decoder_target_tokens, 
    max_decoder_seq_length=max_decoder_len, 
    result_saved_path= test_target_path+TYPE.GUI)
    print(decoded_sentence)
    # generator = NodeTreeGenerator(rule=2)
    # compiler = Compiler(path.DATASET2_DSL_MAPPING_JSON_FILE, rule=2)
    # tree = compiler.dsl_to_node_tree(test_target_path+TYPE.GUI)
    # html = compiler.node_tree_to_html(test_target_path+TYPE.HTML, 'BN')