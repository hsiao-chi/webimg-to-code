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
            # 'data_folder': path.DATASET1_ROWCOL_ATTRIBUTE_GUI,
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


def get_encoder_config(target_type=1):
    if target_type == 1:
        return {
            'direct_part': 5,
            'data_folder': path.DATASET1_FULL_YOLO_NOISE_TXT,
            # 'data_folder': path.DATASET1_FULL_YOLO_POSITION_TXT,
            'testing_data_folder': path.DATASET1_TESTING_SEQ2SEQ_POSITION_NOISE_TXT,
            # 'testing_data_folder': path.DATASET1_TESTING_SEQ2SEQ_POSITION_TXT,
            'class_mode': False,
            'token_list': [],
        }
    elif target_type == 2:
        return {
            'direct_part': 5,
            'data_folder': path.DATASET1_ATTRIBUTE_YOLO_POSITION_TXT,
            'testing_data_folder': path.DATASET1_TESTING_SEQ2SEQ_ATTR_POSITION_TXT,
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