import random
from util import get_random_text
from dslModel import Node, Attribute
from config import *
MAX_DEPTH = 6
MAX_EACH_LAYER_NODE_NUM = 5


def generateNodeTree(parent_node, depth) -> Node:
    node = None
    random_num = random.randrange(1, MAX_EACH_LAYER_NODE_NUM)
    if parent_node.key == NodeKey.list.value:
        random_num = MAX_EACH_LAYER_NODE_NUM
    for num_node in range(random_num):
        # print("depth: ", depth, "  num_node: ", num_node, "/", random_num, "  parent: ", parent_node.key)
        node = generateNode(parent_node, depth)
        # print("choice: ", node.key)
        if depth >=MAX_DEPTH or node.key in [key.value for key in list(LeafKey)]:
           pass
        else:
            node = generateNodeTree(node, depth+1)
        parent_node.add_child(node)
    return parent_node


def generateNode(parent_node, depth) -> Node:
    pool = []
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

    # print(parent_node.key, depth, pool)

    key_enum = random.choice(pool)
    attribute = Attribute(None, None, None)
    if key_enum in GENERATE_RULE['attributes']['none']:
        pass
    else:
        if key_enum in GENERATE_RULE['attributes']['have_font_color_node']:
            color =  random.choice(list(Color)).value
            attribute.set_font_color(color)
        if key_enum in GENERATE_RULE['attributes']['have_bg_color_node']:
            color =  random.choice(list(Color)).value
            attribute.set_bg_color(color)
        if key_enum in GENERATE_RULE['attributes']['have_context_node']:
            context = get_random_text(5, 0)
            attribute.set_context(context)

    return Node(key_enum.value, parent_node, attribute, depth)




if __name__ == "__main__":
    root = Node(RootKey.body.value, None, Attribute(None, None, None))
    tree = generateNodeTree(root, 0)
    tree.show()
