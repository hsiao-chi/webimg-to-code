from nodeTreeGenerator import generateNode, generateNodeTree
from nodeModel import Node, Attribute
from compiler import Compiler
from config import *
import os

TYPE_TXT = '.txt'
TYPE_GUI = '.gui'
TYPE_HTML = '.html'

if __name__ == "__main__":
    # for i in range(2):
        # root = Node(RootKey.body.value, None, Attribute(None, None, None))
        # tree = generateNodeTree(root, 0)
        # compiler = Compiler(Path.web_dsl_mapping_json.value, tree)
        # compiler.node_tree_to_dsl(Path.origin_gui.value+str(i)+TYPE_GUI)
        # compiler.node_tree_to_dsl(Path.row_col_gui.value+str(i)+TYPE_GUI, True)

    data_file_length = len(os.listdir(Path.origin_gui.value))
    # for i in range(data_file_length):
    compiler = Compiler(Path.web_dsl_mapping_json.value)
    tree = compiler.dsl_to_node_tree(Path.origin_gui.value+str(0)+TYPE_GUI)
    print(tree.show())
    html = compiler.node_tree_to_html(Path.origin_html.value+str(0)+TYPE_HTML, str(0))
    print(html)

