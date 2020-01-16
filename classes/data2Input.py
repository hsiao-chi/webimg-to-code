import os
import os.path
import numpy as np
import general.path as path
import general.dataType as TYPE
from general.util import createFolder, read_file, write_file
from PIL import Image


def decoder_tokens_list_to_dict(decoder_token_list: list) -> dict:
    return {e: i for i, e in enumerate(decoder_token_list)}


def to_Seq2Seq_input(encoder_file_folder, decoder_file_folder, encoder_config, decoder_token_list: list, data_num=None):
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
    for i in range(num_total_data):
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


def get_attribute_data(annotation_line, input_shape, tokens_dict, max_attributes=4, proc_img=True ):
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    attrs = ['START']+line[1:]+['EOS']
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2
    image_data = 0
    print('scale: {}, iw: {}, ih: {}, nw: {}, nh: {}, dx: {}, dy: {}'.format(scale, iw, ih, nw, nh, dx, dy))
    if proc_img:
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)/255.

    attributes_input_data = np.zeros((max_attributes, len(tokens_dict)))
    if len(attrs) > 0:
        if len(attrs) > max_attributes:
            attrs = attrs[:max_attributes]
        for i, attr in enumerate(attrs):
            print(i, attr)
            attributes_input_data[i, tokens_dict[attr]] = 1
    return image_data, attributes_input_data 


def attributes_data_generator(annotation_lines, batch_size, input_shape, tokens_list):
    tokens_dict = decoder_tokens_list_to_dict(tokens_list)
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        decoder_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
                image, attributes = get_attribute_data(annotation_lines[i], input_shape, tokens_dict)
                image_data.append(image)
                decoder_data.append(attributes)
                i = (i+1) % n
        
        decoder_output_data = np.roll(decoder_data, -1, axis=1)
        decoder_output_data[:, -1] = 0
        decoder_output_data[:, -1, tokens_dict['EOS']] = 1

        image_data = np.array(image_data)
        decoder_data = np.array(decoder_data)

        return image_data, decoder_data, decoder_output_data
