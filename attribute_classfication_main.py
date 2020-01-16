# from classes.model.attributeClassfication import attribute_classification_train_model, attribute_classfication_training
from classes.data2Input import attributes_data_generator
from general.util import read_file, write_file
from classes.get_configs import get_attribute_encoder_config, get_attribute_decoder_config
import general.path as path
import general.dataType as TYPE

if __name__ == "__main__":

    encoder_config = get_attribute_encoder_config(1)
    decoder_config = get_attribute_decoder_config(1)
    # token_list = get_attribute_decoder_config(2)['token_list']
    lines = read_file(decoder_config['data_path'], 'splitlines')
    # new_lines=[]
    # for line in lines:
    #     line = line.split()
    #     attrs =  [token_list[int(i)] for i in line[1:]]
    #     new_lines.append([line[0]]+attrs)
    # write_file(new_lines, path.DATASET1_ELEMENT_FOLDER+'text-attr-labels-lab.txt', 2)

#    train_model = attribute_classification_train_model(10)
    image_data, decoder_data, decoder_output_data = attributes_data_generator(lines, 2, (20,20), decoder_config['token_list'])
    print('--{}---\n{}'.format('image_data', image_data))
    print('--{}---\n{}'.format('decoder_data', decoder_data))
    print('--{}---\n{}'.format('decoder_output_data', decoder_output_data))


    