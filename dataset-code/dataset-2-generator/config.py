from enum import Enum
from general.node.nodeEnum import RootKey, NodeKey, LeafKey
MAX_EACH_LAYER_NODE_NUM = 4
MAX_DEPTH = 6

GENERATE_RULE = {
    "attributes": {
        "none": [RootKey.body, NodeKey.row, NodeKey.col, NodeKey.list],
        "have_font_color_node": [LeafKey.text, LeafKey.title],
        "have_bg_color_node": [LeafKey.div, NodeKey.button],
        "have_size_node": [NodeKey.button, LeafKey.title],
        "have_content_node": [LeafKey.text, LeafKey.title],
    },
    "enabled_children": {
        # RootKey.body.value: [NodeKey.row, NodeKey.absolute],
        RootKey.body.value: [NodeKey.row],
        NodeKey.button.value: [LeafKey.text],
        NodeKey.row.value: [NodeKey.col],
        # NodeKey.col.value: [NodeKey.row, NodeKey.button, NodeKey.list, LeafKey.text, LeafKey.title],
        NodeKey.col.value: [NodeKey.row],
        # NodeKey.list.value: [NodeKey.button, LeafKey.text, LeafKey.title],
        # NodeKey.absolute.value: [NodeKey.row, NodeKey.list],
    },
    "enabled_parent": {
        NodeKey.button.value: [NodeKey.col, NodeKey.list],
        NodeKey.row.value: [RootKey.body, NodeKey.absolute],
        NodeKey.col.value: [NodeKey.row],
        NodeKey.list.value: [NodeKey.col, NodeKey.absolute],
        NodeKey.absolute.value: [RootKey.body],
    },
    "enabled_brothers": {
        "same": [NodeKey.list],
    },
    "children_quantity_limit": {
        NodeKey.button.value: 1,
        NodeKey.list.value: MAX_EACH_LAYER_NODE_NUM,
    },
    "children_group": {
        # NodeKey.button.value: [LeafKey.icon, LeafKey.text],
        NodeKey.col.value: [LeafKey.title, LeafKey.text, NodeKey.button]
    },
    "disabled_layer": {
        "last_1": [NodeKey.row, NodeKey.absolute],
        "last": list(NodeKey)
    },
}


def getEnumList(enums):
    return [key.value for key in enums]
