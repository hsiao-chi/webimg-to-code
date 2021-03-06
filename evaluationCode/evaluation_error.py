import numpy as np
from general.util import write_file
import json

class Eva_error:
    record_template = {
        'config': {
            'input_path': None,
            'target_path': None,
            'model_path': None,
            'num_of_data': None,
            'start_index': None,
        },
        'error_history': []
        ,
        'final_error_score': {
            'the_number_of_len_nonequal_data': 0,
            # 'sum_error': 0,
            # 'sum_len': 1,
            'error_score': 0
        }
    }
    def __init__(self, start_index, input_path, target_path, predit_model_path):
        self.record_template['config']['input_path']=input_path
        self.record_template['config']['target_path']=target_path
        self.record_template['config']['model_path']=predit_model_path
        self.record_template['config']['start_index']=start_index
        self.record_template['final_error_score']['the_number_of_len_nonequal_data']=0
        self.record_template['error_history']=[]

        self.each_error = np.zeros(1)
        # self.each_len = np.zeros(1)

    def cal_error(self, predited_sequence: list, original_sequence:list):
        error=0
        len_origin = len(original_sequence)
        len_pred = len(predited_sequence)
        if len_origin != len_pred:
            error = 1
            self.record_template['final_error_score']['the_number_of_len_nonequal_data']+=1

        else:
            error_s = 0
            for idx in range(len_origin):
                if predited_sequence[idx] != original_sequence[idx]:
                    error_s+=1
            
            error = error_s/len_origin

        self.each_error = np.append(self.each_error, error)
        # self.each_len = np.append(self.each_len, max_seq_len)

        return error

    def get_final_error(self, save_path: None):
        print('self.each_error', self.each_error)
        # print('self.each_len', self.each_len)
        sum_error = np.sum(self.each_error)
        len_data = np.size(self.each_error)-1
        error_score = sum_error/len_data
        self.record_template['config']['num_of_data']= len_data
        # self.record_template['final_error_score']['sum_error']= sum_error
        # self.record_template['final_error_score']['sum_len']= sum_len
        self.record_template['final_error_score']['error_score']= error_score
        # self.record_template['final_error_score']['the_number_of_high_error(0.3)_data']= len(self.record_template['error_history'])
        
        if save_path:
            with open(save_path, 'a+') as file:
                file.write(json.dumps(self.record_template))
                file.write('\n')
        return error_score