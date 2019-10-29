from nodeTreeGenerator import generateNode, generateNodeTree
from general.node.nodeModel import Node, Attribute
import general.path as path
import general.dataType as TYPE
from compiler import Compiler

import os

if __name__ == "__main__":
    # for i in range(2):
        # root = Node(RootKey.body.value, None, Attribute(None, None, None))
        # tree = generateNodeTree(root, 0)
        # compiler = Compiler(Path.web_dsl_mapping_json.value, tree)
        # compiler.node_tree_to_dsl(Path.origin_gui.value+str(i)+TYPE_GUI)
        # compiler.node_tree_to_dsl(Path.row_col_gui.value+str(i)+TYPE_GUI, True)

    data_file_length = len(os.listdir(path.DATASET2_ORIGIN_GUI))
    # for i in range(data_file_length):
    compiler = Compiler(path.DATASET2_DSL_MAPPING_JSON_FILE)
    tree = compiler.dsl_to_node_tree(path.DATASET2_ORIGIN_GUI+str(1)+TYPE.GUI)
    print(tree.show())
    html = compiler.node_tree_to_html(path.DATASET2_ORIGIN_HTML+str(1)+TYPE.HTML, str(1))
    print(html)

