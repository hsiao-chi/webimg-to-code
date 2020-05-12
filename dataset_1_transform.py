from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Font_color, Bg_color
import general.path as path
import general.dataType as TYPE
from general.util import write_file, read_file, copy_files, createFolder, replace_file_value
from datasetCode.dataset_2_generator.generateRule import getRule
from datasetCode.dataset_2_generator.nodeTreeGenerateClass import NodeTreeGenerator
from datasetCode.dataset_2_generator.compiler import Compiler
from datasetCode.data_transform.web_to_screenshot import webscreenshoot
from datasetCode.data_transform.transform_to_row_col_position import convert_to_position_and_rowcol_img
from datasetCode.data_transform.tag_for_yolo import manual_class_tag_from_file, ManualTagClass, to_yolo_training_file, yolo_position_with_noise_generator
from datasetCode.assests.yolo_class_lists import pix2code_full_classes as buttonList
from datasetCode.data_transform.create_attrcate_dataset import create_attribute_classfication_dataset
from environment.environment import DATASET, DATASET_ANOTHER
import cv2
from classes.get_configs import get_encoder_config
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image

import os
if __name__ == "__main__":
    list1 = os.listdir(path.DATASET1_ORIGIN_PNG)
    num_total_data = len(list1)

    ''' 
    =================================================================================
    --------------- full yolo position to yolo training datafile --------------------
    =================================================================================
    '''
    # createFolder(path.DATASET3_TRAINSET_YOLO_TRAIN_TXT)
    # to_yolo_training_file(path.DATASET3_TRAINSET_ORIGIN_LIGHT_PNG, 
    # path.DATASET3_TRAINSET_ATTR_YOLO_POSITION_TXT, 500,
    # path.DATASET3_TRAINSET_YOLO_TRAIN_TXT+"500\\data3_attr_yolo_500"+TYPE.TXT)
    # to_yolo_training_file(path.DATASET3_TRAINSET_ORIGIN_LIGHT_PNG, 
    # path.DATASET3_TRAINSET_ATTR_YOLO_POSITION_TXT_PADDING, 500,
    # path.DATASET3_TRAINSET_YOLO_TRAIN_TXT+"500\\data3_attr_yolo_500_padding"+TYPE.TXT)

    # to_yolo_training_file(path.DATASET3_TRAINSET_ORIGIN_LIGHT_PNG, 
    # path.DATASET3_TRAINSET_ATTR_YOLO_POSITION_TXT_PADDING_2, 500,
    # path.DATASET3_TRAINSET_YOLO_TRAIN_TXT+"500\\data3_attr_yolo_500_padding_2"+TYPE.TXT)

    # to_yolo_training_file(path.DATASET3_TRAINSET_ORIGIN_LIGHT_PNG, 
    # path.DATASET3_TRAINSET_ATTR_YOLO_POSITION_TXT_INPUT_PADDING, 500,
    # path.DATASET3_TRAINSET_YOLO_TRAIN_TXT+"500\\data3_attr_yolo_500_input_padding"+TYPE.TXT)

    ''' 
    =================================================================================
    ----------------------------- Manual tag yolo class -----------------------------
    =================================================================================
    '''

    for i in range(700, 800):
        root = tk.Tk()
        app = ManualTagClass(root, buttonList, path.DATASET1_ORIGIN_PNG + str(i) + TYPE.IMG,
                             path.DATASET1_ROWCOL_YOLO_POSITION_TXT + str(i) + TYPE.TXT, path.DATASET1_FULL_YOLO_POSITION_TXT + str(i)+TYPE.TXT)
        root.mainloop()

        if app.is_close():
            break

    ''' 
    =================================================================================
    --------------------------------- 忘了幹啥子的 ----------------------------------
    =================================================================================
    '''
    # detectionList, rolColImg = convert_to_position_and_rowcol_img(path.DATASET1_ORIGIN_PNG + str(i)+ TYPE.IMG, path.DATASET1_ROWCOL_YOLO_POSITION_TXT+str(i)+TYPE.TXT, path.DATASET1_ROWCOL_PNG+str(i)+TYPE.IMG, True)
    # if i % 100 == 0:
    #     print(i)

    # print("================ ", str(i), "start ======================")
    # value, interrupt = manual_class_tag_from_file(path.DATASET1_ORIGIN_PNG+ str(i) + TYPE.IMG, path.DATASET1_ROWCOL_YOLO_POSITION_TXT + str(i) + TYPE.TXT)
    # print(value)
    # if interrupt:
    #     print("\n======｡ﾟヽ(ﾟ´Д`)ﾉﾟ｡ ", str(i), "interrupt ｡ﾟヽ(ﾟ´Д`)ﾉﾟ｡ ======")

    #     break
    # write_file(value, path.DATASET1_FULL_YOLO_POSITION_TXT + str(i)+TYPE.TXT, 2)

    ''' 
    =================================================================================
    ------------------------------- CHECK DATASET ERROR -----------------------------
    =================================================================================
    '''

    # for i in range(600):
    #     labels = read_file(path.DATASET1_FULL_YOLO_POSITION_TXT+str(i)+TYPE.TXT, 'splitlines')
    #     for label in labels:
    #         c = label.split()[0]
    #         if int(c) > 6:
    #             print('你就是個問題: ', i, '  值:', c)
    #     print('now: ', str(i)) if i % 50 == 0 else None

    ''' 
    =================================================================================
    ------------- Full Yolo position -- to -- Attribute Yolo position ---------------
    =================================================================================
    '''
    # createFolder(path.DATASET3_TRAINSET_NOISE_ATTR_YOLO_POSITION_TXT)
    # bg_color = [None, None, Bg_color.primary.value, Bg_color.dark.value,
    #             Bg_color.success.value, Bg_color.warning.value, Bg_color.danger.value, 
    #             None, None,None, None,None, None, None]
    # text_color = [Font_color.dark.value, Font_color.dark.value, Font_color.white.value,
    #               Font_color.primary.value, Font_color.white.value, Font_color.white.value, Font_color.white.value,
    #               Font_color.primary.value, Font_color.success.value, Font_color.danger.value,
    #               Font_color.primary.value, Font_color.success.value, Font_color.danger.value, None]
    # classes = [[0, 7, 8, 9], [1, 10, 11, 12], [2, 3, 4, 5, 6], [13]]
    # for i in range(2500):
    #     new_labels = []
    #     labels = read_file(
    #         path.DATASET3_TRAINSET_NOISE_FULL_YOLO_POSITION_TXT+str(i)+TYPE.TXT, 'splitlines')
    #     for label in labels:
    #         c = int(label.split()[0])
    #         position = label.split()[1:]
    #         new_label = []
    #         # new_label.append(str(min(c, 2)))
    #         for nc in range(len(classes)):
    #             if c in classes[nc]:
    #                 break 
    #         new_label.append(str(nc))
    #         new_label += position
    #         # new_label.append(text_color[c])
    #         # if c >= 2:
    #         #     new_label.append(bg_color[c])
    #         attrs = [a for a in [text_color[c], bg_color[c]] if a]
    #         new_label += attrs
    #         new_labels.append(new_label)
    #     write_file(new_labels, path.DATASET3_TRAINSET_NOISE_ATTR_YOLO_POSITION_TXT+str(i)+TYPE.TXT, 2)


    ''' 
    =================================================================================
    --------------- origin-SDL Dataset transfor to row-col-element SDL --------------
    =================================================================================
    '''

    # pix_file_names = read_file('E:\\projects\\NTUST\\webimg-to-code\\datasetCode\\data_transform\\assest\\pix2code_filenames.txt', 'splitlines' )
    # for i, file_name in enumerate(pix_file_names):
    #     sdl = read_file(path.PIX2CODE_ORIGIN_DATASET+file_name+TYPE.GUI, 'noSplit')
    #     sdl = sdl.replace('small-title', 'title')
    #     sdl = sdl.replace(',', '')
    #     sdl = sdl.split()
    #     write_file(sdl, path.DATASET1_ORIGIN_GUI+str(i)+TYPE.GUI, 1)

    ''' 
    =================================================================================
    ------ origin-SDL Dataset transfor to row-col-attribute-subElement SDL ----------
    =================================================================================
    '''

    # pix_file_names = read_file('E:\\projects\\NTUST\\webimg-to-code\\datasetCode\\data_transform\\assest\\pix2code_filenames.txt', 'splitlines' )
    # for i, file_name in enumerate(pix_file_names):
    #     sdl = read_file(path.PIX2CODE_ORIGIN_DATASET+file_name+TYPE.GUI, 'noSplit')
    #     sdl = sdl.replace('small-title', 'title')
    #     sdl = sdl.replace(',', '')
    #     sdl = sdl.replace('header', 'row')
    #     sdl = sdl.replace('single', 'col')
    #     sdl = sdl.replace('double', 'col')
    #    new_positions_folderce('quadruple', 'col')
    #     sdl = sdl.replace('text', 'text [ text-dark ]')
#    new_positions_folder.replace('title', 'title [ text-dark ]')
    #     sdl = sdl.replace('btn-active', 'btn [ text-white bg-primary ]')
    #     sdl = sdl.replace('btn-inactive', 'btn [ text-primary bg-dark ]')
    #     sdl = sdl.replace('btn-green', 'btn [ text-white bg-success ]')
    #     sdl = sdl.replace('btn-orange', 'btn [ text-white bg-warning ]')
    #     sdl = sdl.replace('btn-red', 'btn [ text-white bnew_positions_folder  #     sdl = sdl.split()
    #     write_file(sdl, path.DATASET1_ROWCOL_ATTRIBUTE_GUI+str(i)+TYPE.GUI, 1)

    ''' 
    =================================================================================
    ---------------------- FULL YOLO positions add noise_positions_folder-----------
    =================================================================================
    '''

    # yolo_position_with_noise_generator(
    #     yolo_position_folder=path.DATASET3_TRAINSET_ATTR_YOLO_POSITION_TXT,
    #     # gui_folder=path.DATASET3_TRAINSET_ORIGIN_GUI,
    #     new_positions_folder=path.DATASET3_TRAINSET_NOISE_ATTR_YOLO_POSITION_TXT,
    #     # new_gui_folder=path.DATASET3_TRAINSET_NOISE_ORIGIN_GUI,
    #     data_num=500,
    #     save_origin_file=True,
    #     resort=True)
    ''' 
    =================================================================================
    ---------------------- COPY FILES for TestingDataset ----------------------------
    =================================================================================
    '''
    # origin_start = 400
    # origin_end = 599
    # target_start = 300
    # clean = False
    # copy_files(path.DATASET3_FULL_YOLO_POSITION_TXT, origin_start, origin_end, TYPE.TXT,
    #            path.DATASET3_TRAINSET_FULL_YOLO_POSITION_TXT, target_start, TYPE.TXT, clean_target_folder=clean)
    # copy_files(path.DATASET3_FULL_YOLO_POSITION_TXT_PADDING, origin_start, origin_end, TYPE.TXT,
    #            path.DATASET3_TRAINSET_FULL_YOLO_POSITION_TXT_PADDING, target_start, TYPE.TXT, clean_target_folder=clean)
    # copy_files(path.DATASET3_FULL_YOLO_POSITION_TXT_PADDING_20, origin_start, origin_end, TYPE.TXT,
    #            path.DATASET3_TRAINSET_FULL_YOLO_POSITION_TXT_PADDING_2, target_start, TYPE.TXT, clean_target_folder=clean)
    # copy_files(path.DATASET3_FULL_YOLO_POSITION_TXT_INPUT_PADDING, origin_start, origin_end, TYPE.TXT,
    #            path.DATASET3_TRAINSET_FULL_YOLO_POSITION_TXT_INPUT_PADDING, target_start, TYPE.TXT, clean_target_folder=clean)
    
    # copy_files(path.DATASET3_ATTR_YOLO_POSITION_TXT, origin_start, origin_end, TYPE.TXT,
    #            path.DATASET3_TRAINSET_ATTR_YOLO_POSITION_TXT, target_start, TYPE.TXT, clean_target_folder=clean)
    # copy_files(path.DATASET3_ATTR_YOLO_POSITION_TXT_PADDING, origin_start, origin_end, TYPE.TXT,
    #            path.DATASET3_TRAINSET_ATTR_YOLO_POSITION_TXT_PADDING, target_start, TYPE.TXT, clean_target_folder=clean)
    # copy_files(path.DATASET3_ATTR_YOLO_POSITION_TXT_PADDING_20, origin_start, origin_end, TYPE.TXT,
    #            path.DATASET3_TRAINSET_ATTR_YOLO_POSITION_TXT_PADDING_2, target_start, TYPE.TXT, clean_target_folder=clean)
    # copy_files(path.DATASET3_ATTR_YOLO_POSITION_TXT_INPUT_PADDING, origin_start, origin_end, TYPE.TXT,
    #            path.DATASET3_TRAINSET_ATTR_YOLO_POSITION_TXT_INPUT_PADDING, target_start, TYPE.TXT, clean_target_folder=clean)

    # copy_files(path.DATASET3_ORIGIN_LIGHT_PNG, origin_start, origin_end, TYPE.IMG,
    #            path.DATASET3_TRAINSET_ORIGIN_LIGHT_PNG, target_start, TYPE.IMG, clean_target_folder=clean)

    # copy_files(path.DATASET3_ORIGIN_GUI, origin_start, origin_end, TYPE.GUI,
    #            path.DATASET3_TRAINSET_ORIGIN_GUI, target_start, TYPE.GUI, clean_target_folder=clean)
    # copy_files(path.DATASET1_FULL_YOLO_NOISE_ORIGIN_GUI, 0, 499, TYPE.GUI,
    #            path.DATASET1_FULL_YOLO_NOISE_ORIGIN_GUI, 2000, TYPE.GUI, clean_target_folder=False)


    ''' 
    =================================================================================
    ----------------- CREATE Attribute classification dataset -----------------------
    =================================================================================
    '''
        
    # createFolder(path.DATASET3_ELEMENT_PNG_PADDING_20)
    # create_attribute_classfication_dataset(
    #     path.DATASET3_ATTR_YOLO_POSITION_TXT_PADDING_20, path.DATASET3_ORIGIN_LIGHT_PNG, 
    #     path.DATASET3_ELEMENT_PNG_PADDING_20, path.DATASET3_ELEMENT_FOLDER+'attr-labels-balance-padding20'+TYPE.TXT, 
    #     path.DATASET3_ELEMENT_FOLDER+'record-balance-padding20'+TYPE.TXT,
    #     get_encoder_config(5)['token_list'], 
    #     element_start_index=0, file_start_index=0, file_num=400, balance=True, 
    #     initial_each_element=[0,0,0,0], proportion=[4.,4.,6., 1.])

    ''' 
    =================================================================================
    ----------------- Replace Dataset environment in file ---------------------------
    =================================================================================
    '''

    # replace_file_value(
    #     path.DATASET3_ELEMENT_FOLDER+'attr-labels-button_lab'+TYPE.TXT, 
    #     path.DATASET3_ELEMENT_FOLDER+'attr-labels-button'+TYPE.TXT,
    #     DATASET_ANOTHER, DATASET
    #     )

    # for name in ['attr_yolo_500_input_padding', 'attr_yolo_500_padding', 'attr_yolo_500_padding_2', 'full_yolo_500', 'full_yolo_500_input_padding', 'full_yolo_500_padding', 'full_yolo_500_padding_2']:
    #     replace_file_value(
    #         path.DATASET3_TRAINSET_YOLO_TRAIN_TXT+'500\\data3_'+name+TYPE.TXT, 
    #         path.DATASET3_TRAINSET_YOLO_TRAIN_TXT+'500\\data3_'+name+'_lab'+TYPE.TXT,
    #         DATASET, DATASET_ANOTHER
    #         )
    # replace_file_value(
    #     path.DATASET1_YOLO_TRAIN_DATA+'pix2code_attr_yolo_500_padding_2'+TYPE.TXT, 
    #     path.DATASET1_YOLO_TRAIN_DATA+'pix2code_attr_yolo_500_padding_2_lab'+TYPE.TXT,
    #     DATASET, DATASET_ANOTHER
    #     )


    ''' 
    =================================================================================
    ------------ Transform YOLO Training Dataset to Simple Classes ------------------
    =================================================================================
    '''

    # origin_data= path.DATASET1_YOLO_TRAIN_DATA+'pix2code_full_yolo_500'+TYPE.TXT
    # lines = read_file(origin_data, 'splitlines')
    # with open(path.DATASET1_YOLO_TRAIN_DATA+'pix2code_attr_yolo_500'+TYPE.TXT, 'a+') as f:
    #     for line in lines:
    #         line = line.split()
    #         new_line = [line[0]]
    #         for position in line[1:]:
    #             position = position.split(',')
    #             position[4] = str(min(int(position[4]), 2))
    #             new_line.append(','.join(position))
    #         f.write('{}\n'.format(' '.join(new_line)))
        

    ''' 
    =================================================================================
    ------------ Replace position of row-col to full-yolo ---------------------------
    =================================================================================
    '''


    # for i in [102, 104, 105, 107, 108, 109, 111, 112, 113, 114, 117, 118, 120, 122, 125]:
    #     full_position = read_file(path.DATASET3_FULL_YOLO_POSITION_TXT+str(i)+TYPE.TXT, 'splitlines' )
    #     row_col_position = read_file(path.DATASET3_ROWCOL_YOLO_POSITION_TXT+str(i)+TYPE.TXT, 'splitlines' )
    #     if len(full_position) == len(row_col_position):
    #         new=[]
    #         for j in range(len(full_position)):
    #             full = full_position[j].split()
    #             row_col = row_col_position[j].split()
    #             new.append([full[0]]+row_col[1:])
    #         write_file(new, path.DATASET3_FULL_YOLO_POSITION_TXT+str(i)+TYPE.TXT, dataDim=2)
    #     else:
    #         print('not equal: ', i)
          

    ''' 
    =================================================================================
    ------------ yolo position TEXT_INPUT add padding -------------------------------
    =================================================================================
    '''
    # x-=5/2400, y-=5/2400
    # widh+=10/2400, height+=10/2400
    # createFolder(path.DATASET3_FULL_YOLO_POSITION_TXT_PADDING)
    # for i in range(400, 600):
    #     lines=read_file(path.DATASET3_FULL_YOLO_POSITION_TXT+str(i)+TYPE.TXT, 'splitlines')
    #     new_lines=[]
    #     for line in lines:
    #         line = line.split()
    #         # new_line = [line[0], str(float(line[1])-(5/2400)), str(float(line[2])-(5/1380)), str(float(line[3])+(10/2400)), str(float(line[4])+(10/1380))]+line[5:]

    #         if line[0] == '13':
    #             new_line = [line[0], str(float(line[1])-(5/2400)), str(float(line[2])-(5/1380)), str(float(line[3])+(10/2400)), str(float(line[4])+(10/1380))]+line[5:]
    #         else:
    #             new_line = line
    #         new_lines.append(new_line)
    #     # print(new_lines)
    #     write_file(new_lines, path.DATASET3_FULL_YOLO_POSITION_TXT_INPUT_PADDING+str(i)+TYPE.TXT, dataDim=2)
   

    ''' 
    =================================================================================
    ------------ split attribute element list 'button', anouther --------------------
    =================================================================================
    '''

    # origin_file = read_file(path.DATASET3_ELEMENT_FOLDER+'attr-labels-balance_lab.txt', 'splitlines')
    # button_file = path.DATASET3_ELEMENT_FOLDER+'attr-labels-button_lab.txt'
    # text_file = path.DATASET3_ELEMENT_FOLDER+'attr-labels-text_lab.txt'
    # button_list=[]
    # text_list=[]
    # for line in origin_file:
    #     num = len(line.split())
    #     if num ==2:
    #         text_list.append(line)
    #     elif num == 3:
    #         button_list.append(line)
    
    # write_file(button_list, button_file,join_token="\n")
    # write_file(text_list, text_file, join_token="\n")
