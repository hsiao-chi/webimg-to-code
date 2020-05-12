from classes.get_configs import get_encoder_config, get_decoder_config, get_attribute_encoder_config, get_attribute_decoder_config
from classes.model.layoutGenerator import seq2seq_predit, seq2seq_predit_model
from classes.model.attributeClassfication import attribute_classification_predit_model, attribute_classification_predit

from classes.data2Input import to_Seq2Seq_input, to_Seq2Seq_encoder_input, decoder_tokens_list_to_dict
import os
import os.path
import general.dataType as TYPE
import general.path as path
from general.util import createFolder, showLoss, showAccuracy, read_file, write_file
from keras.models import load_model
from nltk.translate.bleu_score import sentence_bleu
import json
from evaluationCode.bleu import Bleu
from evaluationCode.evaluation_error import Eva_error
import environment.environment as ENV
import cv2
from datasetCode.data_transform.tag_for_yolo import splitImage
from PIL import Image


def attr_classification(encoder_model, decoder_model, img_path, subImg_shape, attr_token_list, detection_list):
    attr_targets = []
    max_dim3_len = 0
    full_image = cv2.imread(img_path)
    for idx, target in enumerate(detection_list):
        sub_img = splitImage(full_image, target)
        sub_img = Image.fromarray(cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB))
        decoded_sentence = attribute_classification_predit(
            encoder_model, decoder_model, sub_img, subImg_shape, attr_token_list, 4, img_input_type='img')
        attr_targets.append(target+decoded_sentence)
        if max_dim3_len < len(target+decoded_sentence):
            max_dim3_len = len(target+decoded_sentence)
        print(decoded_sentence)
    return attr_targets


if __name__ == "__main__":
    BLEU_SCORE = True
    ERROR_SCORE = True
    INPUT_TYPE = 5
    TARGET_TYPE = 4
    seq_en_config = get_encoder_config(INPUT_TYPE)
    seq_de_config = get_decoder_config(TARGET_TYPE)
    attr_en_config = get_attribute_encoder_config(1)
    attr_de_config = get_attribute_decoder_config(1)

    encoder_bidirectional_lstm = False
    predit_data_num = 500
    start_idx = 0
    test_data = False
    # after YOLO
    origin_img_data_folder = path.DATASET3_TRAINSET_ORIGIN_LIGHT_PNG
    input_data_folder = ENV.SELF + r'yolo_detected\data3\arch1_train\010'
    target_data_folder = seq_de_config['testing_data_folder'] if test_data else seq_de_config['data_folder']

    seq_model_path = ENV.SELF + \
        r'assets\2020\seq2seq-data3\remote\attr\1500\normal\model(p0-epoch300).h5'
    attr_class_model_path = r'E:\projects\NTUST\webimg-to-code\assets\2020\attr_class-pix2code\remote\attr_class_model(simple_112_256_2300).h5'

    predit_result_save_path = None
    bleu_file_path = path.EVALUATION_BLEU_SCORE + 'layout_generate_only\\data3\\'
    bleu_file_name = 'Arch2_1500_normal_record.txt'
    error_file_path = path.EVALUATION_ERROR_SCORE + 'layout_generate_only\\data3\\'
    error_file_name = 'Arch2_1500_normal_record.txt'

    encoder_model, decoder_model = seq2seq_predit_model(
        load_model(seq_model_path), bidirectional_lstm=encoder_bidirectional_lstm)
    decoder_target_tokens = decoder_tokens_list_to_dict(
        seq_de_config['token_list'])
    attr_encoder_model, attr_decoder_model = attribute_classification_predit_model(
        load_model(attr_class_model_path))

    max_decoder_len = 300
    if BLEU_SCORE:
        bleu = Bleu(predit_data_num, 0, input_data_folder,
                    target_data_folder, seq_model_path)
    if ERROR_SCORE:
        eva_error = Eva_error(0, input_data_folder,
                              target_data_folder, seq_model_path)

    valid_data_num = predit_data_num
    for idx in range(start_idx, start_idx+predit_data_num):
        input_seqs = read_file(
            input_data_folder+str(idx)+TYPE.TXT, 'splitlines')
        if len(input_seqs) == 0:
            valid_data_num -= 1
            continue
        reference_gui = read_file(
            input_data_folder+str(idx)+TYPE.GUI, 'splitBySpec')

        input_seqs = [seq.split() for seq in input_seqs]
        input_attr_seqs = attr_classification(attr_encoder_model, attr_decoder_model,
                                              img_path=origin_img_data_folder +
                                              str(idx)+TYPE.IMG,
                                              subImg_shape=attr_en_config['input_shape'],
                                              attr_token_list=attr_de_config['token_list'],
                                              detection_list=input_seqs)
        input_seq = to_Seq2Seq_encoder_input(input_attr_seqs, seq_en_config)

        decoded_sentence = seq2seq_predit(encoder_model, decoder_model,
                                          input_seq=input_seq,
                                          decoder_tokens=decoder_target_tokens,
                                          max_decoder_seq_length=max_decoder_len,
                                          result_saved_path=predit_result_save_path +
                                          str(idx)+TYPE.GUI
                                          )

        if BLEU_SCORE:
            bleu.evaluate(decoded_sentence, reference_gui)
        if ERROR_SCORE:
            e = eva_error.cal_error(decoded_sentence, reference_gui)
            print(e)
    if BLEU_SCORE:
        createFolder(bleu_file_path)
        p = bleu_file_path+bleu_file_name
        print(p)
        bleu.save_evaluation(p, valid_data_num)

    if ERROR_SCORE:
        createFolder(error_file_path)
        p = error_file_path+error_file_name
        print(p)
        eva_error.get_final_error(p)
