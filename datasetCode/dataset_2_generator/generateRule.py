from enum import Enum
from general.node.nodeEnum import RootKey, NodeKey, LeafKey, AttributeSet, Font_color, Bg_color


class Operator(Enum):
    none = None
    same = "same"
    whole = "whole"  # any
    more_then = "more_then"
    equal_more_then = "equal_more_then"
    between = "between"
    equal = "equal"
    random = 'random'
    true = True
    false = False


class NodeLayer(Enum):
    first = 'first'
    last = 'last'
    last_1 = 'last_1'
    whole = 'whole'  # any


def getRule(rule=1):
    if rule == 1:
        return rule_1
    elif rule == 2:
        return rule_2
    elif rule == 3:
        return rule_3


rule_1 = {
    "attributes": [AttributeSet.font_color, AttributeSet.bg_color, AttributeSet.content],
    # "use_children_group": Operator.random,
    "max_each_layer_node_num": 5,
    "max_depth": 6,
    RootKey.body.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.row],
        "parents": [],
        "children_brothers": Operator.none,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": None
    },
    NodeKey.row.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.col],
        "parents": [RootKey.body, NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0, 1],
        "children_group": None,
        "children_quantity": {
            "operator": Operator.equal_more_then,
            "value": 2,
        }
    },
    NodeKey.col.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.row],
        "parents": [NodeKey.row],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0, 1],
        "children_group": {
            "enable": Operator.random,
            "nodes": [LeafKey.title, LeafKey.text, NodeKey.button]
        },
        "children_quantity": {
            "operator": Operator.equal,
            "value": 1,
        }
    },
    NodeKey.button.value: {
        "attributes": [False, True, False],
        "children": [],
        "parents": [NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0],
        "children_group": {
            "enable": Operator.true,
            "nodes": [LeafKey.text]
        },
        "children_quantity": {
            "operator": Operator.equal,
            "value": 1,
        }
    },
    LeafKey.text.value: {
        "attributes": [True, False, True],
        "children": [],
        "parents": [NodeKey.col, NodeKey.button],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": None
    },
    LeafKey.title.value: {
        "attributes": [True, False, True],
        "children": [],
        "parents": [NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": None
    },
}

rule_2 = {      # as same as pix2code
    "attributes": [AttributeSet.font_color, AttributeSet.bg_color, AttributeSet.content],
    # "use_children_group": Operator.random,
    "max_each_layer_node_num": 4,
    "max_depth": 6,
    RootKey.body.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.row],
        "parents": [],
        "children_brothers": Operator.none,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": None
    },
    NodeKey.row.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.col],
        "parents": [RootKey.body, NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0, 1],
        "children_group": None,
        "children_quantity": {
            "operator": Operator.equal_more_then,
            "value": 2,
        }
    },
    NodeKey.col.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.row],
        "parents": [NodeKey.row],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0, 1],
        "children_group": {
            "enable": Operator.random,
            "nodes": [LeafKey.title, NodeKey.button, LeafKey.text]
        },
        "children_quantity": {
            "operator": Operator.equal,
            "value": 1,
        }
    },
    LeafKey.button.value: {
        "attributes": [True, False, True],
        "attributes_set": {
            AttributeSet.font_color.value: [Font_color.primary, Font_color.white],
            AttributeSet.bg_color.value: [Bg_color.primary, Bg_color.dark, Bg_color.success, Bg_color.warning, Bg_color.danger],
            "groups": [(Font_color.primary, Bg_color.dark), (Font_color.white, Bg_color.primary),
                       (Font_color.white, Bg_color.success), (Font_color.white, Bg_color.warning), (Font_color.white, Bg_color.danger)],
        },
        "children": [],
        "parents": [NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0],
        "children_group": None,
        "children_quantity": None,
    },
    LeafKey.text.value: {
        "attributes": [True, False, True],
        "attributes_set": {
            AttributeSet.font_color.value: [Font_color.dark],
        },
        "children": [],
        "parents": [NodeKey.col, NodeKey.button],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": None
    },
    LeafKey.title.value: {
        "attributes": [True, False, True],
        "attributes_set": {
            AttributeSet.font_color.value: [Font_color.dark],
        },
        "children": [],
        "parents": [NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": None
    },
}

rule_3 = {      
    "attributes": [AttributeSet.font_color, AttributeSet.bg_color, AttributeSet.content],
    # "use_children_group": Operator.random,
    "max_each_layer_node_num": 3, # (< not <=)
    "max_depth": 6,
    RootKey.body.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.row],
        "parents": [],
        "children_brothers": Operator.none,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": {
            "operator": Operator.equal_more_then,
            "value": 2,
        }
    },
    NodeKey.row.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.col],
        "parents": [RootKey.body, NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0, 1],
        "children_group": None,
        "children_quantity": {
            "operator": Operator.equal_more_then,
            "value": 2,
        }
    },
    NodeKey.col.value: {
        "attributes": [False, False, False],
        "children": [NodeKey.row],
        "parents": [NodeKey.row],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0, 1],
        "children_group": {
            "enable": Operator.random,
            "nodes": [LeafKey.title, LeafKey.text, NodeKey.button]
        },
        "children_quantity": {
            "operator": Operator.between,
            "value": (2,4),
        }
    },
    LeafKey.button.value: {
        "attributes": [True, False, True],
        "attributes_set": {
            AttributeSet.font_color.value: [Font_color.primary, Font_color.white],
            AttributeSet.bg_color.value: [Bg_color.primary, Bg_color.dark, Bg_color.success, Bg_color.warning, Bg_color.danger],
            "groups": [(Font_color.primary, Bg_color.dark), (Font_color.white, Bg_color.primary),
                       (Font_color.white, Bg_color.success), (Font_color.white, Bg_color.warning), (Font_color.white, Bg_color.danger)],
        },
        "children": [],
        "parents": [NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [0],
        "children_group": None,
        "children_quantity": None,
    },
    LeafKey.text.value: {
        "attributes": [True, False, True],
        "attributes_set": {
            AttributeSet.font_color.value: [Font_color.dark, Font_color.success, Font_color.danger, Font_color.primary],
        },
        "children": [],
        "parents": [NodeKey.col, NodeKey.button],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": None
    },
    LeafKey.title.value: {
        "attributes": [True, False, True],
        "attributes_set": {
            AttributeSet.font_color.value: [Font_color.dark, Font_color.success, Font_color.danger, Font_color.primary],
        },
        "children": [],
        "parents": [NodeKey.col],
        "children_brothers": Operator.whole,
        "disabled_reciprocal_layer": [],
        "children_group": None,
        "children_quantity": None
    },
}