from classes.get_configs import get_encoder_config, get_decoder_config
from classes.model.layoutGenerator import seq2seq_predit, seq2seq_predit_model, seq2seq_train_model, seq2seq_training, SEQ2SEQ_EPOCHES
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

    encoder_config = get_encoder_config(INPUT_TYPE)
    decoder_config = get_decoder_config(TARGET_TYPE)

    print(encoder_config['data_folder'])
    list1 = os.listdir(encoder_config['data_folder'])
    num_total_data = len(list1)

    createFolder(path.CLASS_SEQ2SEQ_PREDIT_GUI_PATH + str(SEQ2SEQ_EPOCHES))
    createFolder(path.CLASS_SEQ2SEQ_MODEL_PATH + str(SEQ2SEQ_EPOCHES))
    createFolder(path.CLASS_SEQ2SEQ_WEIGHT + str(SEQ2SEQ_EPOCHES))

    encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len = to_Seq2Seq_input(
        encoder_config['data_folder'], decoder_config['data_folder'], encoder_config, decoder_config['token_list'])

    _, _, num_input_token = encoder_input_data.shape
    _, _, num_target_token = decoder_input_data.shape

    # seq2seq_training_model = seq2seq_train_model(
    #     num_input_token, num_target_token, weight_path='E:\\projects\\NTUST\\webimg-to-code\\assets\\seq2seq-pix2code-full-rowcolAttrElement\\weight\\400\\seq2seq-weights00400.h5')
    seq2seq_training_model = seq2seq_train_model(
        num_input_token, num_target_token)

    final_model_path = path.CLASS_SEQ2SEQ_MODEL_PATH+ str(SEQ2SEQ_EPOCHES)+'\\model'+TYPE.H5
    seq2seq_training_model = seq2seq_training(seq2seq_training_model, encoder_input_data, decoder_input_data, decoder_target_tokens,
                                              analysis_saved_folder=path.CLASS_SEQ2SEQ_ANALYSIS_PATH+ str(SEQ2SEQ_EPOCHES)+"\\",
                                              checkpoint_folder=path.CLASS_SEQ2SEQ_WEIGHT+ str(SEQ2SEQ_EPOCHES)+"\\",
                                              final_model_saved_path=final_model_path)

    encoder_model, decoder_model = seq2seq_predit_model(
        load_model(final_model_path))

    for i in range(5):
        ii = random.randint(0, num_total_data+1)
        input_seq = encoder_input_data[ii: ii+1]
        print(input_seq)
        decoded_sentence = seq2seq_predit(encoder_model, decoder_model, input_seq,
                                          decoder_target_tokens, max_decoder_len,
                                          result_saved_path=path.CLASS_SEQ2SEQ_PREDIT_GUI_PATH + str(SEQ2SEQ_EPOCHES)+'\\'+str(ii)+TYPE.GUI)
        print('decoded_sentence length: ', ii, len(decoded_sentence))
