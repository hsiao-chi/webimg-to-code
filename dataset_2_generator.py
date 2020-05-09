from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Font_color, Bg_color
import general.path as path
import general.dataType as TYPE
from general.util import write_file, createFolder, read_file
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
    WEB_SCREENSHOOT = False
    RULE = 4
    if RANDOM_GENERATOR:
        rule = getRule(RULE)
        generator = NodeTreeGenerator(rule=RULE)
        for i in range(500,600):
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
        for i in [503, 402, 453, 471,472,512,530,536,552, 589]:
            compiler = Compiler(path.DATASET3_DSL_MAPPING_JSON_FILE, rule=RULE)
            tree = compiler.dsl_to_node_tree(path.DATASET3_ORIGIN_GUI+str(i)+TYPE.GUI)

            html = compiler.node_tree_to_html(path.DATASET3_ORIGIN_HTML+str(i)+TYPE.HTML, str(i))

            [web_img_path] = webscreenshoot([path.DATASET3_ORIGIN_HTML+str(i)+TYPE.HTML], path.DATASET3_ORIGIN_PNG, size=(1600,920), deviceScaleFactor=1.5)
            convert_to_position_and_rowcol_img(web_img_path,
                                        path.DATASET3_ROWCOL_YOLO_POSITION_TXT + str(i) + TYPE.TXT, path.DATASET3_ROWCOL_IMG + str(i) + TYPE.IMG)

            # break

    ''' 
    =================================================================================
    ------------ html: remove bg-dark of INPUT, save new file -----------------------
    =================================================================================
    '''
    # createFolder(path.DATASET3_ORIGIN_LIGHT_HTML)
    # createFolder(path.DATASET3_ORIGIN_LIGHT_PNG)
    # for i in range(400, 600):
    #     origin = read_file(path.DATASET3_ORIGIN_HTML+str(i)+TYPE.HTML, 'noSplit')
    #     origin=origin.replace('bg-secondary', '')
    #     write_file(origin, path.DATASET3_ORIGIN_LIGHT_HTML+str(i)+TYPE.HTML, dataDim=0)
    #     [web_img_path] = webscreenshoot([path.DATASET3_ORIGIN_LIGHT_HTML+str(i)+TYPE.HTML], path.DATASET3_ORIGIN_LIGHT_PNG, size=(1600,920), deviceScaleFactor=1.5)

