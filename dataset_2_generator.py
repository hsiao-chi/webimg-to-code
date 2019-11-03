from general.node.nodeModel import Node, Attribute
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, Font_color, Bg_color
import general.path as path
import general.dataType as TYPE
from datasetCode.dataset_2_generator.generateRule import getRule
from datasetCode.dataset_2_generator.nodeTreeGenerateClass import NodeTreeGenerator
from datasetCode.dataset_2_generator.compiler import Compiler
from datasetCode.data_transform.web_to_screenshot import webscreenshoot

import os

if __name__ == "__main__":
    rule = getRule()
    generator = NodeTreeGenerator()
    for i in range(4,5):
        root = Node(RootKey.body.value, None, Attribute(rule["attributes"], rule[RootKey.body.value]["attributes"]))
        tree = generator.generateNodeTree(root, 0)
        compiler = Compiler(path.DATASET2_DSL_MAPPING_JSON_FILE, 1, tree)
        compiler.node_tree_to_dsl(path.DATASET2_ORIGIN_GUI+str(i)+TYPE.GUI)
        compiler.node_tree_to_dsl(path.DATASET2_ROWCOL_GUI+str(i)+TYPE.GUI, True)
        html = compiler.node_tree_to_html(path.DATASET2_ORIGIN_HTML+str(i)+TYPE.HTML, str(i))
        webscreenshoot([path.DATASET2_ORIGIN_HTML+str(i)+TYPE.HTML], path.DATASET2_ORIGIN_PNG)



        # print(html)
    # data_file_length = len(os.listdir(path.DATASET2_ORIGIN_GUI))
    # # for i in range(data_file_length):
    # compiler = Compiler(path.DATASET2_DSL_MAPPING_JSON_FILE, 1)
    # tree = compiler.dsl_to_node_tree(path.DATASET2_ORIGIN_GUI+str(5)+TYPE.GUI)
    # print(tree.show())
    # html = compiler.node_tree_to_html(path.DATASET2_ORIGIN_HTML+str(5)+TYPE.HTML, str(5))
    # print(html)



