from general.node.nodeEnum import Font_color, Bg_color
from general.util import read_file, write_file
import general.dataType as TYPE
import cv2
import json
import numpy as np
from datasetCode.data_transform.tag_for_yolo import splitImage


def to_attrcate_data_old(positions, origin_image, label_list: list)-> dict:
    elements = []
    for position in positions:
        element = {
            'attributes':  [label_list.index(attr) for attr in position[5:]],
            'image': np.uint8(splitImage(origin_image, position)).tolist(),
        }
        elements.append(element)
    return elements


def create_attribute_classfication_dataset_old(positions_folder, image_folder, target_path):
    target_content = {
        'labels': [Font_color.dark.value, Font_color.primary.value, Font_color.white.value,
                   Bg_color.primary.value, Bg_color.dark.value, Bg_color.success.value,
                   Bg_color.warning.value, Bg_color.danger.value],
        'total_data_file_num': 3,
        'training_data_num': 0,
        'testing_data_num': 0,
        'total_data_num': 0,
        'data': [],
    }

    for index in range(target_content['total_data_file_num']):
        img = cv2.imread(image_folder+str(index)+TYPE.IMG)
        read_positions = read_file(
            positions_folder+str(index)+TYPE.TXT, 'splitlines')
        positions = [position.split() for position in read_positions]
        one_file_elements = to_attrcate_data(
            positions, img, target_content['labels'])
        target_content['data'] += one_file_elements

        print('file: ', index) if index % 5 == 0 else None
    target_content['total_data_num'] = len(target_content['data'])
    target_content['training_data_num'] = int(
        len(target_content['data']) * 0.8)
    target_content['testing_data_num'] = int(len(target_content['data']) * 0.2)
    # write_file(target_content, target_path, 'JSON')
    with open(target_path, 'w') as f:
        json.dump(target_content, f)
    print('save json file: ', target_path)
    return target_content


def create_attribute_classfication_dataset(attr_positions_folder, image_folder,
                                           element_folder, target_path, record_path, label_list, element_start_index, file_start_index=0, file_num=1):
    element_index = element_start_index
    with open(target_path, 'a+') as f:
        for file_idx in range(file_start_index, file_start_index+file_num):
            img = cv2.imread(image_folder+str(file_idx)+TYPE.IMG)
            read_positions = read_file(
                attr_positions_folder+str(file_idx)+TYPE.TXT, 'splitlines')
            positions = [position.split() for position in read_positions]
            for position in positions:
                sub_img = splitImage(img, position)
                attributes = position[5:]
                element_file_name = element_folder+str(element_index)+TYPE.IMG
                f.write('{} {}\n'.format(element_file_name, ' '.join( [ str(a) for a in attributes] )))
                cv2.imwrite(element_file_name, sub_img)
                element_index+=1
            print(file_idx) if file_idx %10 == 0 else None

    write_file('number of used file: {}\nnumber of total_elements: {}\n'.format(file_start_index + file_num, element_index), record_path, 0)   
    return element_index
