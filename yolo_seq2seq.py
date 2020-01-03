from classes.get_configs import get_encoder_config, get_decoder_config
from classes.data2Input import to_Seq2Seq_input
import numpy as np
from classes.model.yolo.yolo import YOLO, detect_video 
from PIL import Image
from keras.models import load_model
from classes.model.layoutGenerator import seq2seq_predit_model, seq2seq_predit, SEQ2SEQ_EPOCHES
import general.path as path
import general.dataType as TYPE

def detect_img(yolo, img_name) -> list:
    try:
        image = Image.open(img_name)
    except:
        print('Open Error! Try again!')
    else:
        r_image, r_targets = yolo.detect_image(image)
        r_image.show()
    # yolo.close_session()
    return r_targets

if __name__ == "__main__":
    INPUT_TYPE = 1
    TARGET_TYPE = 3
    USE_EPOCH = 300
    encoder_config = get_encoder_config(INPUT_TYPE)
    decoder_config = get_decoder_config(TARGET_TYPE)

    targets = detect_img(YOLO(), "9.png")
    print(targets)
    input_seq = np.zeros((1, len(targets), len(targets[0])), dtype='float32')
    for i, line in enumerate(targets):
        input_seq[0, i] = line
    print('input_seq:\n', input_seq)
    encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len = to_Seq2Seq_input(
        encoder_config['data_folder'], decoder_config['data_folder'], encoder_config, decoder_config['token_list'])


    final_model_path = path.CLASS_SEQ2SEQ_MODEL_PATH+ str(SEQ2SEQ_EPOCHES)+'\\model'+TYPE.H5
    encoder_model, decoder_model = seq2seq_predit_model(load_model(final_model_path))
    decoded_sentence = seq2seq_predit(encoder_model, decoder_model, 
    input_seq=input_seq, 
    decoder_tokens=decoder_target_tokens, 
    max_decoder_seq_length=max_decoder_len, 
    result_saved_path='9_noice_1_300_2020'+TYPE.GUI)

    print(decoded_sentence)
    