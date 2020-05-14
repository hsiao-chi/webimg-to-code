from nltk.translate.bleu_score import sentence_bleu
import json
import os
import os.path
from general.util import createFolder, write_file

class Bleu:
        
    record_template = {
        'config': {
            'input_path': None,
            'target_path': None,
            'model_path': None,
            'num_of_data': None,
            'start_index': None,
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
    scores = [None]*8
    def __init__(self, total_data_num, start_index, input_path, target_path, predit_model_path):
        self.total_data_num = total_data_num
        self.record_template['config']['input_path']=input_path
        self.record_template['config']['target_path']=target_path
        self.record_template['config']['model_path']=predit_model_path
        self.record_template['config']['num_of_data']=total_data_num
        self.record_template['config']['start_index']=start_index

    def evaluate(self, decoded_sentence, reference_gui):
        for i in range(len(self.labels)):
            scores = sentence_bleu(
                [reference_gui], decoded_sentence, weights=self.weights[i])
            self.record_template['BLEU_SCORE'][self.labels[i]] += scores

    def save_evaluation(self, record_file_name, change_data_num=None):
        data_num = change_data_num if change_data_num else self.total_data_num
        self.record_template['config']['num_of_data']=str(self.total_data_num)+'('+str(data_num)+')'
        for label in self.labels:
            self.record_template['BLEU_SCORE'][label] /= data_num
        print(self.record_template)
        
        with open(record_file_name, 'a+') as file:
            file.write(json.dumps(self.record_template))    