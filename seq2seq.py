from classes.get_configs import get_encoder_config, get_decoder_config
from classes.data2Input import to_Seq2Seq_input, to_Seq2Seq_encoder_input, decoder_tokens_list_to_dict
from classes.model.seq2seq_biLSTM_attention import lstm_attention_model, mapping_gui_skeleton_token_index, lstm_attention_training, generate, SEQ2SEQ_EPOCHES
import os
import os.path
import general.dataType as TYPE
import general.path as path
from general.util import createFolder, showLoss, showAccuracy, read_file, write_file
from keras.models import load_model
import random
from evaluationCode.bleu import Bleu
from evaluationCode.evaluation_error import Eva_error

def positions_to_encoder_input(data_folder, data_num, start_index):
    encoder_input = []
    max_encoder_input_len = 0
    for idx in range(start_index, start_index+data_num):
        temp_lines = read_file(data_folder+str(idx)+TYPE.TXT, 'splitlines')
        max_encoder_input_len = max(max_encoder_input_len, len(temp_lines))
        temp = [temp_line.split() for temp_line in temp_lines]
        encoder_input.append(temp)
    return encoder_input, max_encoder_input_len

def gui_to_decoder_input(data_folder, data_num, start_index):
    decoder_input = []
    max_decoder_input_len = 0
    for idx in range(start_index, start_index+data_num):
        temp = read_file(data_folder+str(idx)+TYPE.GUI,'splitBySpec')
        max_decoder_input_len = max(max_decoder_input_len, len(temp))
        decoder_input.append(temp)
    return decoder_input, max_decoder_input_len

if __name__ == "__main__":
    INPUT_TYPE = 1
    TARGET_TYPE = 3
    # seq_model_type = SeqModelType.encoder_bidirectional_attention.value
    num_encoder_input_vec=5 #5 or 15
    max_encoder_len=50
    max_decoder_len=250
    layer2_lstm = False
    training_data_num = 2
    training_start_idx = 0
    evaluate_data_num = 500
    eva_record_file_path = path.EVALUATION_SEQ2SEQ_EVALUATION+'pix2code\\'
    eva_record_file_name = 'Arch1_test.txt'
    predit_data_nums = [1, 1] # train, test
    # predit_test_data = False
    
    bleu_record_file_path =  path.EVALUATION_BLEU_SCORE + 'layout_generate_only\\pix2code\\'
    bleu_record_file_name = 'Arch1_test.txt'

    error_record_file_path =  path.EVALUATION_ERROR_SCORE + 'layout_generate_only\\pix2code\\'
    error_record_file_name = 'Arch1_test_bn1.txt'
    
    gaussian_noise = None  # None
    early_stoping = False
    TRAINING = True
    PREDIT = True
    EVALUATE = False
    BLEU_SCORE = False
    ERROR_SCORE = False

    encoder_config = get_encoder_config(INPUT_TYPE)
    decoder_config = get_decoder_config(TARGET_TYPE)
    final_model_path = path.CLASS_SEQ2SEQ_MODEL_PATH + str(SEQ2SEQ_EPOCHES)+'\\model'+TYPE.H5
    predit_model_path = final_model_path
    # evaluate_model_path = r'E:\projects\NTUST\webimg-to-code\assets\2020\seq2seq-pix2code\full-rowcolAttrElement\2500\bidirectional-resort-noise\model\300\model.h5'
    evaluate_model_path = final_model_path

    if TRAINING:
        createFolder(path.CLASS_SEQ2SEQ_MODEL_PATH + str(SEQ2SEQ_EPOCHES))
        createFolder(path.CLASS_SEQ2SEQ_WEIGHT + str(SEQ2SEQ_EPOCHES))
        encoder_train_input, max_encoder_input_len  = positions_to_encoder_input(encoder_config['data_folder'], training_data_num, training_start_idx)
        decoder_train_input, max_decoder_input_len = gui_to_decoder_input(decoder_config['data_folder'], training_data_num, training_start_idx)
        # print('main encoder_train_input', max_encoder_input_len, encoder_train_input)
        # print('main decoder_train_input', max_decoder_input_len, decoder_train_input)


        seq2seq_training_model = lstm_attention_model(num_encoder_input_vec=num_encoder_input_vec, 
                max_decoder_output_length=max_decoder_len, 
                output_dict_size=len(decoder_config['token_list'])+1)
        seq2seq_training_model = lstm_attention_training(seq2seq_training_model, 
                encoder_input_list=encoder_train_input,
                decoder_input_list=decoder_train_input,
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                checkpoint_folder=path.CLASS_SEQ2SEQ_WEIGHT +str(SEQ2SEQ_EPOCHES)+"\\",
                analysis_saved_folder=path.CLASS_SEQ2SEQ_ANALYSIS_PATH,
                final_model_saved_path=final_model_path,
                encoder_max_len=max_encoder_len,
                decoder_max_len=max_decoder_len)

    # if EVALUATE:
        # createFolder(eva_record_file_path)
        # evaluate_save_text=''
        # str_model_path = 'evaluated Model path: \n{}'.format(evaluate_model_path)
        # str_training = '\ntraining data path: \n encoder: {}\n decoder: {}'.format(encoder_config['data_folder'], decoder_config['data_folder'])
        # print(str_model_path)
        # print(str_training)
        # encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len = to_Seq2Seq_input(
        #     encoder_config['data_folder'], decoder_config['data_folder'], encoder_config, decoder_config['token_list'], data_num=evaluate_data_num)

        # str_training_result = seq2seq_evaluate(load_model(evaluate_model_path), encoder_input_data,
        #                  decoder_input_data, decoder_target_tokens)
        # str_testing = '\n\ntesting data path: \n encoder: {}\n decoder: {}'.format(
        #     encoder_config['testing_data_folder'], decoder_config['testing_data_folder'])
        # print(str_testing)
        # encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len = to_Seq2Seq_input(
        #     encoder_config['testing_data_folder'], decoder_config['testing_data_folder'], encoder_config, decoder_config['token_list'])

        # str_testing_result = seq2seq_evaluate(load_model(evaluate_model_path), encoder_input_data,
        #                  decoder_input_data, decoder_target_tokens)
        # evaluate_save_text = str_model_path+str_training+str_training_result+str_testing+str_testing_result
        # write_file(evaluate_save_text, eva_record_file_path+eva_record_file_name, dataDim=0)
        
    
    if PREDIT:
        createFolder(path.CLASS_SEQ2SEQ_PREDIT_GUI_PATH + str(SEQ2SEQ_EPOCHES))

        model = load_model(predit_model_path)
        for data_folder, predit_data_num in zip(['data_folder', 'testing_data_folder'], predit_data_nums):
            valid_data_num = predit_data_num

            if BLEU_SCORE:
                bleu = Bleu(predit_data_num, 0, encoder_config[data_folder], decoder_config[data_folder], predit_model_path)
            if ERROR_SCORE:
                eva_error = Eva_error(0, encoder_config[data_folder], decoder_config[data_folder], predit_model_path)
            for i in range(predit_data_num):
                reference_gui = None
                input_seqs = read_file(
                    encoder_config[data_folder]+str(i)+TYPE.TXT, 'splitlines')
                if len(input_seqs)==0:
                    valid_data_num -= 1
                    continue
                
                input_seqs = [seq.split() for seq in input_seqs]
                decoded_sentence = generate(model, 
                encoder_input_list=input_seqs,
                encoder_config=encoder_config,
                max_output_len=max_decoder_len,
                gui_token_dict=decoder_config['token_list'])
                
                # print('decoded_sentence length: ', i, len(decoded_sentence)) if i%50==0 and BLEU_SCORE else None
                print('decoded_sentence length: ', i, decoded_sentence) 

                if BLEU_SCORE:
                    reference_gui = read_file(
                        decoder_config[data_folder]+str(i)+TYPE.GUI, 'splitBySpec')
                    bleu.evaluate(decoded_sentence, reference_gui)
                if ERROR_SCORE:
                    reference_gui = reference_gui if reference_gui else read_file(
                        decoder_config[data_folder]+str(i)+TYPE.GUI, 'splitBySpec')
                    e= eva_error.cal_error(decoded_sentence, reference_gui)
                    print(e)
            if BLEU_SCORE:
                createFolder(bleu_record_file_path)
                p = bleu_record_file_path+bleu_record_file_name
                print(p)
                bleu.save_evaluation(p, valid_data_num)
            
            if ERROR_SCORE:
                createFolder(error_record_file_path)
                p = error_record_file_path+error_record_file_name
                print(p)
                eva_error.get_final_error(p)


    
