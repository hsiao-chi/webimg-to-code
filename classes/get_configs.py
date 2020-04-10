import general.path as path
from general.node.nodeEnum import Font_color, Bg_color

def get_decoder_config(target_type=1):
    if target_type == 1:
        return {
            'data_folder': path.DATASET1_ORIGIN_GUI,
            'token_list': [
                '{', '}',
                'row', 'header', 'single', 'double', 'quadruple',
                'title', 'text',
                'btn-active', 'btn-inactive', 'btn-green', 'btn-orange', 'btn-red',
                'START', 'EOS']}
    elif target_type == 2:
        return {
            'data_folder': path.DATASET1_ROWCOL_ELEMENT_GUI,
            'token_list': [
                '{', '}',
                'row', 'col',
                'title', 'text',
                'btn-active', 'btn-inactive', 'btn-green', 'btn-orange', 'btn-red',
                'START', 'EOS']}
    elif target_type == 3:
        return {
            'data_folder': path.DATASET1_FULL_YOLO_NOISE_GUI,
            # 'testing_data_folder': path.DATASET1_TESTING_SEQ2SEQ_ATTR_GUI,
            'testing_data_folder': path.DATASET1_TESTING_SEQ2SEQ_ATTR_NOISE_GUI,
            'token_list': [
                '{', '}', '[', ']',
                'row', 'col',
                'title', 'text', 'btn',
                'text-white', 'text-primary', 'text-dark',
                'bg-primary', 'bg-dark', 'bg-success', 'bg-warning', 'bg-danger',
                'START', 'EOS']}
    elif target_type == 4:      # dataset3 
        return {
            'data_folder': path.DATASET3_TRAINSET_NOISE_ORIGIN_NO_CONTEXT_GUI,
            # 'testing_data_folder': path.DATASET1_TESTING_SEQ2SEQ_ATTR_GUI,
            'testing_data_folder': path.DATASET3_TESTSET_NOISE_ORIGIN_NO_CONTEXT_GUI,
            'token_list': [
                '{', '}', '[', ']',
                'row', 'col',
                'title', 'text', 'btn', 'text_input',
                'text-white', 'text-primary', 'text-dark', 'text-success', 'text-danger',
                'bg-primary', 'bg-dark', 'bg-success', 'bg-warning', 'bg-danger',
                'START', 'EOS']}


def get_encoder_config(target_type=1):
    if target_type == 1:
        return {
            'direct_part': 5,
            'data_folder': path.DATASET1_FULL_YOLO_NOISE_TXT,
            'testing_data_folder': path.DATASET1_TESTING_SEQ2SEQ_POSITION_NOISE_TXT,
            # 'testing_data_folder': path.DATASET1_TESTING_SEQ2SEQ_POSITION_TXT,
            'class_mode': False,
            'token_list': [],
        }
    elif target_type == 2:
        return {
            'direct_part': 5,
            'data_folder': path.DATASET1_ATTR_YOLO_NOISE_TXT,
            'testing_data_folder': path.DATASET1_TESTING_SEQ2SEQ_ATTR_POSITION_NOISE_TXT,
            'class_mode': False,
            'token_list': [Font_color.dark.value, Font_color.primary.value, Font_color.white.value,
                           Bg_color.primary.value, Bg_color.dark.value, Bg_color.success.value,
                           Bg_color.warning.value, Bg_color.danger.value],
        }
    elif target_type == 3:
        return {
            'direct_part': 5,
            'data_folder': path.DATASET1_ATTRIBUTE_YOLO_POSITION_TXT,
            'testing_data_folder': path.DATASET1_TESTING_SEQ2SEQ_ATTR_POSITION_TXT,
            'class_mode': True,
            'token_list': [
                [Font_color.dark.value, Font_color.primary.value,
                    Font_color.white.value],
                [Bg_color.primary.value, Bg_color.dark.value, Bg_color.success.value,
                 Bg_color.warning.value, Bg_color.danger.value]
            ],
        }
    elif target_type == 4:      # Dataset3 - arch1-seq2seq
        return {
            'direct_part': 5,
            'data_folder': path.DATASET3_TRAINSET_NOISE_FULL_YOLO_POSITION_TXT_PADDING_2,
            'testing_data_folder': path.DATASET3_TESTSET_NOISE_FULL_YOLO_POSITION_TXT_PADDING_2,
            # 'testing_data_folder': path.DATASET1_TESTING_SEQ2SEQ_POSITION_TXT,
            'class_mode': False,
            'token_list': [],
        }
    elif target_type == 5:      # Dataset3 - arch2-seq2seq
        return {
            'direct_part': 5,
            'data_folder': path.DATASET3_TRAINSET_NOISE_ATTR_YOLO_POSITION_TXT_PADDING_2,
            'testing_data_folder': path.DATASET3_TESTSET_NOISE_ATTR_YOLO_POSITION_TXT_PADDING_2,
            'class_mode': False,
            'token_list': [Font_color.dark.value, Font_color.primary.value, Font_color.white.value, Font_color.success.value, Font_color.danger.value, 
                           Bg_color.primary.value, Bg_color.dark.value, Bg_color.success.value,
                           Bg_color.warning.value, Bg_color.danger.value],
        }
    


def get_attribute_encoder_config(target_type=1):
    if target_type == 1:
        return {
            'data_folder': path.DATASET3_ELEMENT_PNG_PADDING_20,
            'input_shape': (74, 224, 3), #(224, 224, 3)
            'num_train': 800,
            'num_valid': 330,
            'num_test': 200,
        }
        

def get_attribute_decoder_config(target_type=1):
    if target_type == 1:
        return {
            'data_path': path.DATASET3_ELEMENT_FOLDER+'attr-labels-text_lab.txt',
            'token_list': [Font_color.dark.value, Font_color.primary.value, Font_color.white.value, Font_color.success.value, Font_color.danger.value,
                           Bg_color.primary.value, Bg_color.dark.value, Bg_color.success.value,
                           Bg_color.warning.value, Bg_color.danger.value, 'START', 'EOS'],
        }
        