from classes.get_configs import get_encoder_config, get_decoder_config
from classes.data2Input import to_Seq2Seq_input
import numpy as np
from classes.model.yolo.yolo import YOLO, detect_video 
from PIL import Image
from keras.models import load_model
from classes.model.layoutGenerator import seq2seq_predit_model, seq2seq_predit, SEQ2SEQ_EPOCHES
import general.path as path
import general.dataType as TYPE
from datasetCode.dataset_2_generator.nodeTreeGenerateClass import NodeTreeGenerator
from datasetCode.dataset_2_generator.compiler import Compiler
from datasetCode.data_transform.web_to_screenshot import webscreenshoot

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
    INPUT_TYPE = 1
    TARGET_TYPE = 3
    USE_EPOCH = 300

    test_image_path = r"test-predit\1706\1706.png"
    test_target_folder = 'E:\\projects\\NTUST\\webimg-to-code\\test-predit\\1706\\'
    test_target_path = test_target_folder + '500-BN'
    test_model_path = r'E:\projects\NTUST\webimg-to-code\assets\2020\seq2seq-pix2code\full-rowcolAttrElement\500\bidirectional-resort-noise\model\300\model.h5'
    bidirectional_lstm = True

    encoder_config = get_encoder_config(INPUT_TYPE)
    decoder_config = get_decoder_config(TARGET_TYPE)

    targets = detect_img(YOLO(), test_image_path)
    print(targets)
    input_seq = np.zeros((1, len(targets), len(targets[0])), dtype='float32')
    for i, line in enumerate(targets):
        input_seq[0, i] = line
    print('input_seq:\n', input_seq)
    encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len = to_Seq2Seq_input(
        encoder_config['data_folder'], decoder_config['data_folder'], encoder_config, decoder_config['token_list'])


    encoder_model, decoder_model = seq2seq_predit_model(load_model(test_model_path), bidirectional_lstm=bidirectional_lstm)
    decoded_sentence = seq2seq_predit(encoder_model, decoder_model, 
    input_seq=input_seq, 
    decoder_tokens=decoder_target_tokens, 
    max_decoder_seq_length=max_decoder_len, 
    result_saved_path= test_target_path+TYPE.GUI)

    print(decoded_sentence)

    generator = NodeTreeGenerator(rule=2)
    compiler = Compiler(path.DATASET2_DSL_MAPPING_JSON_FILE, rule=2)
    tree = compiler.dsl_to_node_tree(test_target_path+TYPE.GUI)
    html = compiler.node_tree_to_html(test_target_path+TYPE.HTML, 'BN')
    # [web_img_path] = webscreenshoot([test_target_path+TYPE.HTML], test_target_folder, (2400,1380))


    