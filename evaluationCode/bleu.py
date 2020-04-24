from nltk.translate.bleu_score import sentence_bleu
import json
import os
import os.path
class Bleu:
        
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
    def __init__(self, total_data_num):
        self.total_data_num = total_data_num

    def evaluate(self, decoded_sentence, reference_gui):
        for i in range(len(self.labels)):
            scores[i] = sentence_bleu(
                [reference_gui], decoded_sentence, weights=self.weights[i])
            self.record_template['BLEU_SCORE'][labels[i]] += scores[i]

    def save_evaluation(record_file_name):
        for label in labels:
            record_template['BLEU_SCORE'][label] /= self.total_data_num
        print(self.record_template)
        
        with open(record_file_name, 'a+') as file:
            file.write(json.dumps(self.record_template))    