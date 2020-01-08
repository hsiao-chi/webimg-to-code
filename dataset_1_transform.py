from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Font_color, Bg_color
import general.path as path
import general.dataType as TYPE
from general.util import write_file, read_file, copy_files
from datasetCode.dataset_2_generator.generateRule import getRule
from datasetCode.dataset_2_generator.nodeTreeGenerateClass import NodeTreeGenerator
from datasetCode.dataset_2_generator.compiler import Compiler
from datasetCode.data_transform.web_to_screenshot import webscreenshoot
from datasetCode.data_transform.transform_to_row_col_position import convert_to_position_and_rowcol_img
from datasetCode.data_transform.tag_for_yolo import manual_class_tag_from_file, ManualTagClass, to_yolo_training_file, yolo_position_with_noise_generator
from datasetCode.assests.yolo_class_lists import pix2code_full_classes as buttonList
import cv2

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

    # to_yolo_training_file(path.DATASET1_ORIGIN_PNG, path.DATASET1_FULL_YOLO_POSITION_TXT, 150, path.DATASETCODE_ASSESTS+"pix2code_full_yolo"+TYPE.TXT)

    ''' 
    =================================================================================
    ----------------------------- Manual tag yolo class -----------------------------
    =================================================================================
    '''

    # for i in range(600,700):
    #     root = tk.Tk()
    #     app = ManualTagClass(root, buttonList, path.DATASET1_ORIGIN_PNG + str(i) + TYPE.IMG,
    #                          path.DATASET1_ROWCOL_YOLO_POSITION_TXT + str(i) + TYPE.TXT, path.DATASET1_FULL_YOLO_POSITION_TXT + str(i)+TYPE.TXT)
    #     root.mainloop()

    #     if app.is_close():
    #         break

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

    # for i in range(400):
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

    # bg_color = [None, None, Bg_color.primary.value, Bg_color.dark.value,
    #             Bg_color.success.value, Bg_color.warning.value, Bg_color.danger.value]
    # text_color = [Font_color.dark.value, Font_color.dark.value, Font_color.white.value,
    #               Font_color.primary.value, Font_color.white.value, Font_color.white.value, Font_color.white.value]
    # for i in range(500, 600):
    #     new_labels = []
    #     labels = read_file(
    #         path.DATASET1_FULL_YOLO_POSITION_TXT+str(i)+TYPE.TXT, 'splitlines')
    #     for label in labels:
    #         c = int(label.split()[0])
    #         position = label.split()[1:]
    #         new_label = []
    #         new_label.append(str(min(c, 2)))
    #         new_label += position
    #         new_label.append(text_color[c])
    #         if c >= 2:
    #             new_label.append(bg_color[c])
    #         new_labels.append(new_label)
    #     write_file(new_labels, path.DATASET1_ATTRIBUTE_YOLO_POSITION_TXT+str(i)+TYPE.TXT, 2)

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
    ---------------------- FULL YOLO positions add noinew_positions_folder-----------
    =================================================================================
    '''

    # yolo_position_with_noise_generator(
    #     path.DATASET1_FULL_YOLO_POSITION_TXT,
    #     path.DATASET1_ROWCOL_ATTRIBUTE_GUI,
    #     path.DATASET1_FULL_YOLO_NOISE_TXT,
    #     path.DATASET1_FULL_YOLO_NOISE_GUI,
    #     resort=True)
    ''' 
    =================================================================================
    ---------------------- COPY FILES for TestingDataset ----------------------------
    =================================================================================
    '''

    copy_files(path.DATASET1_FULL_YOLO_POSITION_TXT, 500, 599, TYPE.TXT,
               path.DATASET1_TESTING_SEQ2SEQ_POSITION_TXT, 0, TYPE.TXT)
    copy_files(path.DATASET1_ROWCOL_ATTRIBUTE_GUI, 500, 599, TYPE.GUI,
               path.DATASET1_TESTING_SEQ2SEQ_ATTR_GUI, 0, TYPE.GUI)
