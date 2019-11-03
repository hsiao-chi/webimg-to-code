from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Font_color, Bg_color
import general.path as path
import general.dataType as TYPE
from datasetCode.dataset_2_generator.generateRule import getRule
from datasetCode.dataset_2_generator.nodeTreeGenerateClass import NodeTreeGenerator
from datasetCode.dataset_2_generator.compiler import Compiler
from datasetCode.data_transform.web_to_screenshot import webscreenshoot
from datasetCode.data_transform.transform_to_row_col_position import convert_to_position_and_rowcol_img

import os

if __name__ == "__main__":
    rule = getRule()
    generator = NodeTreeGenerator()
    for i in range(7,8):
        root = Node(RootKey.body.value, None, Attribute(rule["attributes"], rule[RootKey.body.value]["attributes"]))
        tree = generator.generateNodeTree(root, 0)
        compiler = Compiler(path.DATASET2_DSL_MAPPING_JSON_FILE, 1, tree)
        compiler.node_tree_to_dsl(path.DATASET2_ORIGIN_GUI+str(i)+TYPE.GUI)
        compiler.node_tree_to_dsl(path.DATASET2_ROWCOL_GUI+str(i)+TYPE.GUI, True)
        html = compiler.node_tree_to_html(path.DATASET2_ORIGIN_HTML+str(i)+TYPE.HTML, str(i))
        [web_img_path] = webscreenshoot([path.DATASET2_ORIGIN_HTML+str(i)+TYPE.HTML], path.DATASET2_ORIGIN_PNG)
        convert_to_position_and_rowcol_img(web_img_path,
                                       path.DATASET2_ROWCOL_YOLO_POSITION_TXT + str(i) + TYPE.TXT, path.DATASET2_ROWCOL_IMG + str(i) + TYPE.IMG)

        