import os
import os.path
import numpy as np
import general.path as path
import general.dataType as TYPE
from general.util import createFolder, read_file, write_file
from PIL import Image


def decoder_tokens_list_to_dict(decoder_token_list: list) -> dict:
    return {e: i for i, e in enumerate(decoder_token_list)}

def encoder_tokens_list_to_dict(encoder_token_list, class_mode=False):
    encoder_tokens = None
    if class_mode:
        encoder_tokens = [{e: (i+1) for i, e in enumerate(c)}
                          for c in encoder_token_list]
    else:
        encoder_tokens = {e: i for i, e in enumerate(
            encoder_token_list)}

    print('encoder_tokens', encoder_tokens)
    return encoder_tokens

def to_Seq2Seq_encoder_input(input_seqs: list, encoder_config) -> np.array:
    encoder_tokens = encoder_tokens_list_to_dict(encoder_config['token_list'], encoder_config['class_mode'])
    temp_data = []
    for input_seq in input_seqs:
        data = input_seq[:encoder_config['direct_part']]
        attrs = [0]*len(encoder_tokens)
        for attr in input_seq[encoder_config['direct_part']:]:
            if encoder_config['class_mode']:
                for idx, target_list in enumerate(encoder_tokens):
                    try:
                        attrs[idx] = target_list[attr]
                    except KeyError:
                        pass
            else:
                attrs[encoder_tokens[attr]] = 1
        temp_data.append(data+attrs)

    encoder_input_data = np.zeros((1, len(temp_data), len(encoder_tokens)+encoder_config['direct_part']), dtype='float32')
    for i, data in enumerate(temp_data):
        encoder_input_data[0, i] = data
    return encoder_input_data


def to_Seq2Seq_input(encoder_file_folder, decoder_file_folder, encoder_config, decoder_token_list: list, data_num=None, data_start_idx=0):
    num_total_data = data_num
    if data_num == None:
        list1 = os.listdir(encoder_file_folder)
        num_total_data = len(list1)
    # num_total_data = 1
    decoder_target_tokens = decoder_tokens_list_to_dict(decoder_token_list)
    encoder_direct_part = encoder_config['direct_part']
    encoder_tokens = None
    if encoder_config['class_mode']:
        encoder_tokens = [{e: (i+1) for i, e in enumerate(c)}
                          for c in encoder_config['token_list']]
    else:
        encoder_tokens = {e: i for i, e in enumerate(
            encoder_config['token_list'])}

    print('encoder_tokens', encoder_tokens)

    temp_encoder_all_data = []
    temp_decoder_all_data = []
    max_encoder_len = 0
    max_decoder_len = 0
    for i in range(data_start_idx, data_start_idx+num_total_data):
        input_data = read_file(encoder_file_folder +
                               str(i)+TYPE.TXT, 'splitlines')
        gui = read_file(decoder_file_folder+str(i)+TYPE.GUI, 'splitBySpec')
        # print(gui)
        temp_data = []
        for line in input_data:
            l = line.split()
            data = l[:encoder_direct_part]
            attrs = [0]*len(encoder_tokens)
            for attr in l[encoder_direct_part:]:
                if encoder_config['class_mode']:
                    for idx, target_list in enumerate(encoder_tokens):
                        try:
                            attrs[idx] = target_list[attr]
                        except KeyError:
                            pass
                else:
                    attrs[encoder_tokens[attr]] = 1
            temp_data.append(data+attrs)
        temp_encoder_all_data.append(temp_data)
        if len(temp_data) > max_encoder_len:
            max_encoder_len = len(temp_data)

        temp_decoder_all_data.append(['START']+gui)
        if len(gui)+1 > max_decoder_len:
            max_decoder_len = len(gui)+1

    encoder_input_data = np.zeros((num_total_data, max_encoder_len, len(
        encoder_tokens)+encoder_direct_part), dtype='float32')
    decoder_input_data = np.zeros(
        (num_total_data, max_decoder_len, len(decoder_target_tokens)), dtype='float32')
    for i, (temp_data, gui) in enumerate(zip(temp_encoder_all_data, temp_decoder_all_data)):
        for j, data in enumerate(temp_data):
            encoder_input_data[i, j] = data
        for j, token in enumerate(gui):
            decoder_input_data[i, j, decoder_target_tokens[token]] = 1
        decoder_input_data[i, j+1:, decoder_target_tokens['EOS']] = 1

    return encoder_input_data, decoder_input_data, decoder_target_tokens, max_decoder_len


def preprocess_image(image_path, input_shape, proc_img=True, img_input_type='path', keep_ratio=True) -> np.ndarray:
    if img_input_type == 'path':
        image = Image.open(image_path)
    elif img_input_type == 'img':
        image = image_path
    iw, ih = image.size
    h, w, c = input_shape
    image_data = 0
    if proc_img:
        if keep_ratio:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.
        else:
            image = image.resize((w, h), Image.BICUBIC)
            image_data = np.array(image)/255.
    return image_data

def get_attribute_data(annotation_line, input_shape, tokens_dict, max_attributes=4, proc_img=True ):
    line = annotation_line.split()
    image_data = preprocess_image(line[0], input_shape)
    
    attrs = ['START']+line[1:]+['EOS']
    attributes_input_data = np.zeros((max_attributes, len(tokens_dict)))
    if len(attrs) > 0:
        if len(attrs) > max_attributes:
            attrs = attrs[:max_attributes]
        for i, attr in enumerate(attrs):
            attributes_input_data[i, tokens_dict[attr]] = 1
    return image_data, attributes_input_data 


def attributes_data_generator(annotation_lines, batch_size, input_shape, tokens_list, max_attributes=4):
    tokens_dict = decoder_tokens_list_to_dict(tokens_list)
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        decoder_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, attributes = get_attribute_data(annotation_lines[i], input_shape, tokens_dict, max_attributes=max_attributes)
            image_data.append(image)
            decoder_data.append(attributes)
            i = (i+1) % n
        
        image_data = np.array(image_data)
        decoder_data = np.array(decoder_data)
        
        # print('decoder_data_shape: {}'.format(decoder_data.shape))
        decoder_output_data = np.roll(decoder_data, -1, axis=1)
        decoder_output_data[:, -1] = 0
        decoder_output_data[:, -1, tokens_dict['EOS']] = 1

        # print('image_data_shape: {}'.format(image_data.shape))
        # print('decoder_output_data_shape: {}'.format(decoder_output_data.shape))
        # print('[image_data, decoder_data]_shape: {}'.format([image_data, decoder_data]))
        yield [image_data, decoder_data], decoder_output_data
