from classes.get_configs import get_encoder_config, get_decoder_config
from classes.model.layoutGenerator import seq2seq_predit, seq2seq_predit_model, seq2seq_train_model, seq2seq_training, seq2seq_evaluate, SEQ2SEQ_EPOCHES
from classes.data2Input import to_Seq2Seq_input
import os
import os.path
import general.dataType as TYPE
import general.path as path
from general.util import createFolder, showLoss, showAccuracy
from keras.models import load_model
import random

if __name__ == "__main__":
    INPUT_TYPE = 1
    TARGET_TYPE = 3
    encoder_bidirectional_lstm = False
    training_data_num = 2500
    gaussian_noise = 1 #None
    TRAINING = False
    PREDIT = False
    EVALUATE = False
    BLEU_SCORE = True

    encoder_config = get_encoder_config(INPUT_TYPE)
    decoder_config = get_decoder_config(TARGET_TYPE)
    final_model_path = path.CLASS_SEQ2SEQ_MODEL_PATH + str(SEQ2SEQ_EPOCHES)+'\\model'+TYPE.H5
    predit_model_path = final_model_path
    # evaluate_model_path = r'E:\projects\NTUST\webimg-to-code\assets\2020\seq2seq-pix2code\full-rowcolAttrElement\2500\bidirectional-resort-noise\model\300\model.h5'
    evaluate_model_path = final_model_path

    if TRAINING:
        createFolder(path.CLASS_SEQ2SEQ_MODEL_PATH + str(SEQ2SEQ_EPOCHES))
        createFolder(path.CLASS_SEQ2SEQ_WEIGHT + str(SEQ2SEQ_EPOCHES))

        encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len = to_Seq2Seq_input(
            encoder_config['data_folder'], decoder_config['data_folder'], encoder_config, decoder_config['token_list'], data_num=training_data_num)

        _, _, num_input_token = encoder_input_data.shape
        _, _, num_target_token = decoder_input_data.shape

        seq2seq_training_model = seq2seq_train_model(
            num_input_token, num_target_token, gaussian_noise=gaussian_noise,
            encoder_bidirectional_lstm=encoder_bidirectional_lstm)
        seq2seq_training_model = seq2seq_training(seq2seq_training_model, encoder_input_data, decoder_input_data, decoder_target_tokens,
                                                  analysis_saved_folder=path.CLASS_SEQ2SEQ_ANALYSIS_PATH,
                                                  checkpoint_folder=path.CLASS_SEQ2SEQ_WEIGHT + str(SEQ2SEQ_EPOCHES)+"\\",
                                                  final_model_saved_path=final_model_path,
                                                  initial_epoch=0)

    if PREDIT:
        createFolder(path.CLASS_SEQ2SEQ_PREDIT_GUI_PATH + str(SEQ2SEQ_EPOCHES))
        encoder_model, decoder_model = seq2seq_predit_model(
            load_model(predit_model_path), bidirectional_lstm=encoder_bidirectional_lstm)
        list1 = os.listdir(encoder_config['data_folder'])
        num_total_data = len(list1)
        for i in range(5):
            ii = random.randint(0, num_total_data+1)
            input_seq = encoder_input_data[ii: ii+1]
            print(input_seq)
            decoded_sentence = seq2seq_predit(encoder_model, decoder_model, input_seq,
                                              decoder_target_tokens, max_decoder_len,
                                              result_saved_path=path.CLASS_SEQ2SEQ_PREDIT_GUI_PATH + str(SEQ2SEQ_EPOCHES)+'\\'+str(ii)+TYPE.GUI)
            print('decoded_sentence length: ', ii, len(decoded_sentence))

    if EVALUATE:
        print('evaluated Model path: \n{}'.format(evaluate_model_path))
        print('testing data path: \n encoder: {}\n decoder: {}'.format(encoder_config['testing_data_folder'], decoder_config['testing_data_folder']))
        encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len = to_Seq2Seq_input(
            encoder_config['testing_data_folder'], decoder_config['testing_data_folder'], encoder_config, decoder_config['token_list'])

        seq2seq_evaluate(load_model(evaluate_model_path), encoder_input_data,
                         decoder_input_data, decoder_target_tokens)

