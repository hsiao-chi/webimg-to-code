import random
from util import get_random_text
from dslModel import Node, Attribute
from config import NodeKey, LeafKey, RootKey, Color, have_font_color_attr_node, have_bg_color_attr_node, have_size_attr_node, have_context_attr_node

MAX_DEPTH = 4
MAX_EACH_LAYER_NODE_NUM = 5


def generateNodeTree(parent_node, deep) -> Node:
    node = None
    if deep >=MAX_DEPTH:
        node = generateNode(parent_node, LeafKey)

    else:
        node = generateNode(parent_node, NodeKey)
        generateNodeTree(node, deep+1)
    
    parent_node.add_child(node)
    return node


def generateNode(parent_node, typeKey) -> Node:
    key = None
    if parent_node.key == RootKey.body.value:
        key = NodeKey.row.value
    elif parent_node.key == NodeKey.row.value:
        key = NodeKey.col.value
    else:
        key = random.choice(list(typeKey)).value

    attribute = Attribute(None, None, None)
    if key in [key.value for key in have_font_color_attr_node]:
        color =  random.choice(list(Color)).value
        attribute.set_font_color(color)
    
    if key in [key.value for key in have_bg_color_attr_node]:
        color =  random.choice(list(Color)).value
        attribute.set_bg_color(color)

    if key in [key.value for key in have_context_attr_node]:
        context = get_random_text(5, 0)
        attribute.set_context(context)

    node = Node(key, parent_node, attribute)
    return Node(key, parent_node, attribute)


def generateDSLCode(nodeTree):
    pass


if __name__ == "__main__":
    root = Node(RootKey.body.value, None, None)
    tree = generateNodeTree(root, 0)
    tree.show()
