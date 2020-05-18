from classes.get_configs import get_encoder_config, get_decoder_config
from classes.model.layoutGenerator import seq2seq_predit, seq2seq_predit_model, seq2seq_train_model, seq2seq_training, seq2seq_evaluate, SEQ2SEQ_EPOCHES, SeqModelType
from classes.data2Input import to_Seq2Seq_input, to_Seq2Seq_encoder_input, decoder_tokens_list_to_dict
import os
import os.path
import general.dataType as TYPE
import general.path as path
from general.util import createFolder, showLoss, showAccuracy, read_file, write_file
from keras.models import load_model
import random
from evaluationCode.bleu import Bleu
from evaluationCode.evaluation_error import Eva_error

if __name__ == "__main__":
    INPUT_TYPE = 1
    TARGET_TYPE = 1
    seq_model_type = SeqModelType.encoder_bidirectional_attention.value
    layer2_lstm = False
    training_data_num = 2500
    evaluate_data_nums = [500, 100]
    eva_record_file_path = path.EVALUATION_SEQ2SEQ_EVALUATION+'pix2code\\'
    eva_record_file_name = 'Arch1-0_2500_ebiLSTM-attention-128_record.txt'
    predit_data_nums = [500, 100] # train, test
    predit_start_idx = [0,0] # train, test
    # predit_test_data = False
    
    bleu_record_file_path =  path.EVALUATION_BLEU_SCORE + 'layout_generate_only\\2020-04\\pix2code\\'
    bleu_record_file_name = 'Arch1-0_2500_ebiLSTM-attention-128_record.txt'

    error_record_file_path =  path.EVALUATION_ERROR_SCORE + 'layout_generate_only\\pix2code\\'
    error_record_file_name = 'Arch1-0_2500_ebiLSTM-attention-128_record.txt'
    
    gaussian_noise = 1  # None
    early_stoping = False
    TRAINING = True
    EVALUATE = True
    PREDIT = True
    BLEU_SCORE = True
    ERROR_SCORE = True

    encoder_config = get_encoder_config(INPUT_TYPE)
    decoder_config = get_decoder_config(TARGET_TYPE)
    final_model_path = path.CLASS_SEQ2SEQ_MODEL_PATH + str(SEQ2SEQ_EPOCHES)+'\\model'+TYPE.H5
    predit_model_path = final_model_path
    # evaluate_model_path = r'E:\projects\NTUST\webimg-to-code\assets\2020\seq2seq-pix2code\full-rowcolAttrElement\2500\bidirectional-resort-noise\model\300\model.h5'
    evaluate_model_path = final_model_path
    pretrained_weight_path = None

    if TRAINING:
        createFolder(path.CLASS_SEQ2SEQ_MODEL_PATH + str(SEQ2SEQ_EPOCHES))
        createFolder(path.CLASS_SEQ2SEQ_WEIGHT + str(SEQ2SEQ_EPOCHES))

        encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len = to_Seq2Seq_input(
            encoder_config['data_folder'], decoder_config['data_folder'], encoder_config, decoder_config['token_list'], data_num=training_data_num)

        _, _, num_input_token = encoder_input_data.shape
        _, _, num_target_token = decoder_input_data.shape

        seq2seq_training_model = seq2seq_train_model(
            num_input_token, num_target_token, gaussian_noise=gaussian_noise,
            model_type=seq_model_type, layer2_lstm=layer2_lstm)
        seq2seq_training_model = seq2seq_training(seq2seq_training_model, encoder_input_data, decoder_input_data, decoder_target_tokens,
                                                  analysis_saved_folder=path.CLASS_SEQ2SEQ_ANALYSIS_PATH,
                                                  checkpoint_folder=path.CLASS_SEQ2SEQ_WEIGHT +
                                                  str(SEQ2SEQ_EPOCHES)+"\\",
                                                  final_model_saved_path=final_model_path,
                                                  initial_epoch=0,
                                                  enable_early_stopping=early_stoping)

    if EVALUATE:
        createFolder(eva_record_file_path)
        evaluate_save_text=''
        str_model_path = 'evaluated Model path: \n{}'.format(evaluate_model_path)
        str_training = '\ntraining data path: \n encoder: {}\n decoder: {}'.format(encoder_config['data_folder'], decoder_config['data_folder'])
        print(str_model_path)
        print(str_training)
        encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len = to_Seq2Seq_input(
            encoder_config['data_folder'], decoder_config['data_folder'], encoder_config, decoder_config['token_list'], data_num=evaluate_data_nums[0])

        str_training_result = seq2seq_evaluate(load_model(evaluate_model_path), encoder_input_data,
                         decoder_input_data, decoder_target_tokens)
        str_testing = '\n\ntesting data path: \n encoder: {}\n decoder: {}'.format(
            encoder_config['testing_data_folder'], decoder_config['testing_data_folder'])
        print(str_testing)
        encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len = to_Seq2Seq_input(
            encoder_config['testing_data_folder'], decoder_config['testing_data_folder'], encoder_config, decoder_config['token_list'], data_num=evaluate_data_nums[1])

        str_testing_result = seq2seq_evaluate(load_model(evaluate_model_path), encoder_input_data,
                         decoder_input_data, decoder_target_tokens)
        evaluate_save_text = str_model_path+str_training+str_training_result+str_testing+str_testing_result
        createFolder(eva_record_file_path)
        write_file(evaluate_save_text, eva_record_file_path+eva_record_file_name, dataDim=0)
        
    
    if PREDIT:
        createFolder(path.CLASS_SEQ2SEQ_PREDIT_GUI_PATH + str(SEQ2SEQ_EPOCHES))
        decoder_target_tokens = decoder_tokens_list_to_dict(decoder_config['token_list'])
        max_decoder_len = 300
        encoder_model, decoder_model = seq2seq_predit_model(
            load_model(predit_model_path), model_type=seq_model_type, layer2_lstm=layer2_lstm)
        # data_folder = 'testing_data_folder' if predit_test_data else 'data_folder'
        for data_folder, predit_data_num, start_idx in zip(['data_folder', 'testing_data_folder'], predit_data_nums, predit_start_idx):
            valid_data_num = predit_data_num

            if BLEU_SCORE:
                bleu = Bleu(predit_data_num, start_idx, encoder_config[data_folder], decoder_config[data_folder], predit_model_path)
            if ERROR_SCORE:
                eva_error = Eva_error(start_idx, encoder_config[data_folder], decoder_config[data_folder], predit_model_path)
            for i in range(start_idx, start_idx+predit_data_num):
                reference_gui = None
                input_seqs = read_file(
                    encoder_config[data_folder]+str(i)+TYPE.TXT, 'splitlines')
                if len(input_seqs)==0:
                    valid_data_num -= 1
                    continue
                input_seqs = [seq.split() for seq in input_seqs]
                # print('encoder input seq len', len(input_seqs))
                input_seq = to_Seq2Seq_encoder_input(input_seqs, encoder_config)
                # print(input_seq)
                decoded_sentence = seq2seq_predit(encoder_model, decoder_model,
                                                input_seq=input_seq, decoder_tokens=decoder_target_tokens,
                                                max_decoder_seq_length=max_decoder_len,
                                                # result_saved_path=path.CLASS_SEQ2SEQ_PREDIT_GUI_PATH + str(SEQ2SEQ_EPOCHES)+'\\'+str(i)+TYPE.GUI
                                                )
                
                print('decoded_sentence length: ', i, decoded_sentence) if i%50==0 and BLEU_SCORE else None
                # print('decoded_sentence length: ', i, ' '.join(decoded_sentence)) 

                if BLEU_SCORE:
                    reference_gui = read_file(
                        decoder_config[data_folder]+str(i)+TYPE.GUI, 'splitBySpec')
                    bleu.evaluate(decoded_sentence, reference_gui)
                if ERROR_SCORE:
                    reference_gui = reference_gui if reference_gui else read_file(
                        decoder_config[data_folder]+str(i)+TYPE.GUI, 'splitBySpec')
                    e = eva_error.cal_error(decoded_sentence, reference_gui)
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
                res = eva_error.get_final_error(p)
                print(res)



    
