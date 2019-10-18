import random
from util import get_random_text
from dslModel import Node, Attribute
from config import *


def generateNodeTree(parent_node, depth) -> Node:
    node = None
    group_pool = []
    same_brother = False
    node_num = random.randrange(1, MAX_EACH_LAYER_NODE_NUM)
    if parent_node.key in GENERATE_RULE["children_quantity_limit"]:
        node_num = GENERATE_RULE["children_quantity_limit"][parent_node.key]

    if parent_node.key in GENERATE_RULE["children_group"]:
        if random.choice([False, True]) or (MAX_DEPTH-depth) <=2:
            node_num = len(GENERATE_RULE["children_group"][parent_node.key])
            group_pool = GENERATE_RULE["children_group"][parent_node.key]
    
    if parent_node.key in getEnumList(GENERATE_RULE["enabled_brothers"]["same"]):
        same_brother = True

    for index in range(node_num):
        # print("depth: ", depth, "  num_node: ", index, "/", node_num, "  parent: ", parent_node.key, "  group_pool: ", group_pool)
        if len(group_pool) > 0:
            node = generateNode(parent_node, depth+1, group_pool[index].value)
        elif same_brother and index == 0:
            node = generateNode(parent_node, depth+1)
            same_brother = node.key
        elif same_brother:
            node = generateNode(parent_node, depth+1, same_brother)
        else:
            node = generateNode(parent_node, depth+1)

        # print("choice: ", node.key)
        if depth >=MAX_DEPTH or node.key in getEnumList(list(LeafKey)):
           pass
        else:
            node = generateNodeTree(node, depth+1)
        parent_node.add_child(node)
    return parent_node


def generateNode(parent_node, depth, assigned_key=None) -> Node:
    pool = []
    key = None
    if assigned_key:
        key = assigned_key
    else:
        if parent_node.key in GENERATE_RULE['enabled_children']:
            pool = GENERATE_RULE['enabled_children'][parent_node.key]
        
        if MAX_DEPTH - depth == 1:
            for node_enum in GENERATE_RULE['disabled_layer']['last_1']:
                if node_enum in pool:
                    pool.remove(node_enum)
        elif MAX_DEPTH - depth == 0:
            for node_enum in GENERATE_RULE['disabled_layer']['last']:
                if node_enum in pool:
                    pool.remove(node_enum)
        key= random.choice(pool).value
    # print(parent_node.key, depth, pool)
    attribute = Attribute(None, None, None)
    if key in getEnumList(GENERATE_RULE['attributes']['none']):
        pass
    else:
        if key in getEnumList(GENERATE_RULE['attributes']['have_font_color_node']):
            color =  random.choice(list(Color)).value
            attribute.set_font_color(color)
        if key in getEnumList(GENERATE_RULE['attributes']['have_bg_color_node']):
            color =  random.choice(list(Color)).value
            attribute.set_bg_color(color)
        if key in getEnumList(GENERATE_RULE['attributes']['have_context_node']):
            context = get_random_text(5 if parent_node.key == NodeKey.button.value else 20 )
            attribute.set_context("\"" + context + "\"")

    return Node(key, parent_node, attribute, depth)




if __name__ == "__main__":
    root = Node(RootKey.body.value, None, Attribute(None, None, None))
    tree = generateNodeTree(root, 0)
    print(tree.toDSL())
    print("=================================")
    print(tree.to_row_col_DSL())
