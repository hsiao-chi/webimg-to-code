from classes.get_configs import get_encoder_config, get_decoder_config
from classes.model.layoutGenerator import seq2seq_predit, seq2seq_predit_model
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

if __name__ == "__main__":
    BLEU_SCORE = True
    ERROR_SCORE = True
    INPUT_TYPE = 5
    TARGET_TYPE = 4
    encoder_bidirectional_lstm = False
    encoder_config = get_encoder_config(INPUT_TYPE)
    decoder_config = get_decoder_config(TARGET_TYPE)
    predit_data_num = 500
    start_idx = 0
    test_data = False
    input_data_folder = ENV.SELF+ r'yolo_detected\data3\arch1_train\010'
    target_data_folder = decoder_config['testing_data_folder'] if test_data else decoder_config['data_folder']
    predit_model_path = ENV.SELF+ r'assets\2020\seq2seq-data3\remote\attr\1500\normal\model(p0-epoch300).h5'
    predit_result_save_path = None
    bleu_file_path = path.EVALUATION_BLEU_SCORE + 'layout_generate_only\\data3\\'
    bleu_file_name = 'Arch2_1500_normal_record.txt'
    error_file_path = path.EVALUATION_ERROR_SCORE + 'layout_generate_only\\data3\\'
    error_file_name = 'Arch2_1500_normal_record.txt'

    
    encoder_model, decoder_model = seq2seq_predit_model(
        load_model(predit_model_path), bidirectional_lstm=encoder_bidirectional_lstm)
    decoder_target_tokens = decoder_tokens_list_to_dict(decoder_config['token_list'])
    max_decoder_len = 300
    if BLEU_SCORE:
        bleu = Bleu(predit_data_num, 0, input_data_folder, target_data_folder, predit_model_path)
    if ERROR_SCORE:
        eva_error = Eva_error(0, input_data_folder, target_data_folder, predit_model_path)

    valid_data_num = predit_data_num
    for idx in range(start_idx, start_idx+predit_data_num):
        input_seqs= read_file(
                   input_data_folder+str(idx)+TYPE.TXT, 'splitlines')
        if len(input_seqs)==0:
            valid_data_num -= 1
            continue
        reference_gui = read_file(
            input_data_folder+str(idx)+TYPE.GUI, 'splitBySpec')

        input_seqs = [seq.split() for seq in input_seqs]
        input_seq = to_Seq2Seq_encoder_input(input_seqs, encoder_config)
            
        decoded_sentence = seq2seq_predit(encoder_model, decoder_model,
                                                input_seq=input_seq, 
                                                decoder_tokens=decoder_target_tokens,
                                                max_decoder_seq_length=max_decoder_len,
                                                result_saved_path=predit_result_save_path+str(idx)+TYPE.GUI
                                                )

        if BLEU_SCORE:
            bleu.evaluate(decoded_sentence, reference_gui)
        if ERROR_SCORE:
            e= eva_error.cal_error(decoded_sentence, reference_gui)
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
        
       
   
