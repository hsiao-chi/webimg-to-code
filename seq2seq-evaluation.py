from classes.get_configs import get_encoder_config, get_decoder_config
from classes.model.layoutGenerator import seq2seq_predit, seq2seq_predit_model
from classes.data2Input import to_Seq2Seq_input
import os
import os.path
import general.dataType as TYPE
import general.path as path
from general.util import createFolder, showLoss, showAccuracy, read_file, write_file
from keras.models import load_model
from nltk.translate.bleu_score import sentence_bleu
import json

if __name__ == "__main__":
    BLEU_SCORE = True
    INPUT_TYPE = 5
    TARGET_TYPE = 4
    encoder_bidirectional_lstm = False
    testing = True
    data_folder = 'testing_data_folder' if testing else 'data_folder'
    encoder_config = get_encoder_config(INPUT_TYPE)
    decoder_config = get_decoder_config(TARGET_TYPE)
    predit_model_path = r'E:\projects\NTUST\webimg-to-code\assets\2020\seq2seq-data3\remote\attr\1500\normal\model(p0-epoch300).h5'
    record_file_name = path.EVALUATION_BLEU_SCORE + \
        'layout_generate_only\\data3\\Arch2_1500_normal_record.txt'
    history_file_name = path.EVALUATION_BLEU_SCORE + \
        'layout_generate_only\\data3\\Arch2_1500_normal_history.txt'

    if BLEU_SCORE:
        DATA_NUM = 500
        START_IDX = 0
        encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len = to_Seq2Seq_input(
            encoder_config[data_folder], decoder_config[data_folder], encoder_config, decoder_config['token_list'], data_num=DATA_NUM, data_start_idx=START_IDX)
        createFolder(path.EVALUATION_BLEU_SCORE + 'layout_generate_only\\data3\\')
        encoder_model, decoder_model = seq2seq_predit_model(
            load_model(predit_model_path), bidirectional_lstm=encoder_bidirectional_lstm)

        record_template = {
            'config': {
                'input_path': encoder_config[data_folder],
                'target_path': decoder_config[data_folder],
                'model_path': predit_model_path,
                'num_of_data': DATA_NUM,
                'start_index': START_IDX,
            },
            'BLEU_SCORE': {
                'individual_1_gram': 0,
                'individual_2_gram': 0,
                'individual_3_gram': 0,
                'individual_4_gram': 0,
                'cumulative_1_gram': 0,
                'cumulative_2_gram': 0,
                'cumulative_3_gram': 0,
                'cumulative_4_gram': 0,
            }
        }

        weights = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
                   (1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
        labels = ['individual_1_gram', 'individual_2_gram', 'individual_3_gram', 'individual_4_gram',
                  'cumulative_1_gram', 'cumulative_2_gram', 'cumulative_3_gram', 'cumulative_4_gram']
        with open(history_file_name, 'a+') as file:
            file.write('labels list: {}\n'.format(labels))

            for idx in range(DATA_NUM):
                scores = [None]*8
                input_seq = encoder_input_data[idx: idx+1]
                reference_gui = read_file(
                    decoder_config[data_folder]+str(START_IDX+idx)+TYPE.GUI, 'splitBySpec')

                decoded_sentence = seq2seq_predit(encoder_model, decoder_model, input_seq,
                                                  decoder_target_tokens, max_decoder_len,
                                                  result_saved_path=None)

                for i in range(len(labels)):
                    scores[i] = sentence_bleu(
                        [reference_gui], decoded_sentence, weights=weights[i])
                    record_template['BLEU_SCORE'][labels[i]] += scores[i]

                print('decoded_sentence length: ',
                      START_IDX+idx, len(decoded_sentence))
                file.write('file_idx:{}, scores: {}\n'.format(
                    START_IDX+idx, scores))

        for label in labels:
            record_template['BLEU_SCORE'][label] /= DATA_NUM

        print(record_template)

        # write_file(record_template, record_file_name, 'JSON')
        with open(record_file_name, 'a+') as file:
            file.write(json.dumps(record_template))
   
