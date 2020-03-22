from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Font_color, Bg_color
import general.path as path
import general.dataType as TYPE
from general.util import write_file, createFolder
from datasetCode.dataset_2_generator.generateRule import getRule
from datasetCode.dataset_2_generator.nodeTreeGenerateClass import NodeTreeGenerator
from datasetCode.dataset_2_generator.compiler import Compiler
from datasetCode.data_transform.web_to_screenshot import webscreenshoot
from datasetCode.data_transform.transform_to_row_col_position import convert_to_position_and_rowcol_img
from datasetCode.data_transform.tag_for_yolo import manual_class_tag_from_file

import os
if __name__ == "__main__":

    RANDOM_GENERATOR = False
    SKELETON_TO_HTML_ONLY = False
    WEB_SCREENSHOOT = True
    RULE = 4
    if RANDOM_GENERATOR:
        rule = getRule(RULE)
        generator = NodeTreeGenerator(rule=RULE)
        for i in range(100,200):
            print(i)
            root = Node(RootKey.body.value, None, Attribute(rule["attributes"], rule[RootKey.body.value]["attributes"]))
            tree = generator.generateNodeTree(root, 0)
            compiler = Compiler(path.DATASET3_DSL_MAPPING_JSON_FILE, rule=RULE, node_tree=tree)
            # compiler = Compiler(path.DATASET2_DSL_MAPPING_JSON_FILE, rule=2)
            compiler.node_tree_to_dsl(path.DATASET3_ORIGIN_GUI+str(i)+TYPE.GUI)
            # compiler.node_tree_to_dsl(path.DATASET2_ROWCOL_GUI+str(i)+TYPE.GUI, True)
            compiler.node_tree_to_dsl(path.DATASET3_ORIGIN_NO_CONTEXT_GUI+str(i)+TYPE.GUI, with_context=False)
            # tree = compiler.dsl_to_node_tree(path.DATASET2_ORIGIN_GUI+str(i)+TYPE.GUI)
            # print(tree.show())
            html = compiler.node_tree_to_html(path.DATASET3_ORIGIN_HTML+str(i)+TYPE.HTML, str(i))
            [web_img_path] = webscreenshoot([path.DATASET3_ORIGIN_HTML+str(i)+TYPE.HTML], path.DATASET3_ORIGIN_PNG, size=(1600,920), deviceScaleFactor=1.5)
            convert_to_position_and_rowcol_img(web_img_path,
                                        path.DATASET3_ROWCOL_YOLO_POSITION_TXT + str(i) + TYPE.TXT, path.DATASET3_ROWCOL_IMG + str(i) + TYPE.IMG)

        # value = manual_class_tag_from_file(path.DATASET2_ORIGIN_PNG+ str(2) + TYPE.IMG, path.DATASET2_ROWCOL_YOLO_POSITION_TXT + str(2) + TYPE.TXT)  
        # print(value)
        # write_file(value, path.DATASET2_YOLO_POSITION_TXT + str(2)+TYPE.TXT, 2)

    if SKELETON_TO_HTML_ONLY:
        rule = getRule(2)
        # generator = NodeTreeGenerator(rule=2)
        skeleton_file_path = r'E:\projects\NTUST\webimg-to-code\test-predit\dataset2-mapping-adjust\test.gui'
        target_file_path = r'E:\projects\NTUST\webimg-to-code\test-predit\dataset2-mapping-adjust\\test.html'
        compiler = Compiler(path.DATASET2_DSL_MAPPING_JSON_FILE, rule=2)
        tree = compiler.dsl_to_node_tree(skeleton_file_path)
        html = compiler.node_tree_to_html(target_file_path, 'test')

    if WEB_SCREENSHOOT:
            # [web_img_path] = webscreenshoot([path.DATASET2_ORIGIN_HTML+str(1)+TYPE.HTML], r'E:\projects\NTUST\webimg-to-code\\', size=(1200,690), deviceScaleFactor=2)
        for i in [0, 1, 4, 8, 10, 11, 12, 15, 22, 24, 27, 31, 35, 36, 37, 38, 39, 40, 42, 44, 46, 48, 51, 53, 55, 56, 57, 60, 61, 62, 63, 67, 69, 70, 71, 72, 73, 74, 76, 77, 79, 80, 81, 86, 87, 92, 95, 97, 98, 99, 100, 102, 104, 105, 107, 108, 109, 111, 112, 113, 114, 117, 118, 120, 122, 125, 126, 128, 130, 131, 136, 144, 149, 152, 154, 155, 157, 159, 160, 165, 166, 167, 168, 171, 174, 178, 184, 185, 187, 189, 190, 191, 196]:
            compiler = Compiler(path.DATASET3_DSL_MAPPING_JSON_FILE, rule=RULE)
            tree = compiler.dsl_to_node_tree(path.DATASET3_ORIGIN_GUI+str(i)+TYPE.GUI)

            html = compiler.node_tree_to_html(path.DATASET3_ORIGIN_HTML+str(i)+TYPE.HTML, str(i))

            [web_img_path] = webscreenshoot([path.DATASET3_ORIGIN_HTML+str(i)+TYPE.HTML], path.DATASET3_ORIGIN_PNG, size=(1600,920), deviceScaleFactor=1.5)
            convert_to_position_and_rowcol_img(web_img_path,
                                        path.DATASET3_ROWCOL_YOLO_POSITION_TXT + str(i) + TYPE.TXT, path.DATASET3_ROWCOL_IMG + str(i) + TYPE.IMG)

            # break
