from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Font_color, Bg_color
import general.path as path
import general.dataType as TYPE
from general.util import write_file, read_file
from datasetCode.dataset_2_generator.generateRule import getRule
from datasetCode.dataset_2_generator.nodeTreeGenerateClass import NodeTreeGenerator
from datasetCode.dataset_2_generator.compiler import Compiler
from datasetCode.data_transform.web_to_screenshot import webscreenshoot
from datasetCode.data_transform.transform_to_row_col_position import convert_to_position_and_rowcol_img
from datasetCode.data_transform.tag_for_yolo import manual_class_tag_from_file, ManualTagClass, to_yolo_training_file
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

    # to_yolo_training_file(path.DATASET1_ORIGIN_PNG, path.DATASET1_FULL_YOLO_POSITION_TXT, 150, path.DATASETCODE_ASSESTS+"pix2code_full_yolo"+TYPE.TXT)


    for i in range(400,500):
        root = tk.Tk()
        app = ManualTagClass(root, buttonList, path.DATASET1_ORIGIN_PNG + str(i) + TYPE.IMG,
                             path.DATASET1_ROWCOL_YOLO_POSITION_TXT + str(i) + TYPE.TXT, path.DATASET1_FULL_YOLO_POSITION_TXT + str(i)+TYPE.TXT)
        root.mainloop()

        if app.is_close():
            break

    # -----------------------------------
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
